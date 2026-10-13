"""Reproducible regression diagnostics and isolated, bounded mixture ablations."""
from collections import Counter, defaultdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import shutil
import time
import torch
import torch.nn.functional as F
from codexa_workspace.assets import resolve_input
from codexa_workspace.chat_evaluation import evaluate_loaded, make_suite
from llm_architecture.checkpointing import read_native_checkpoint, file_sha256
from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
from llm_data.chat_dataset import IndexedChatDataset
from llm_inference.generate import GenerationConfig, generate_sequences
from llm_tokenizer.sft import ChatMessage, format_chat_prompt
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.session import atomic_document, SessionBudget

def paired_changes(before: list[dict], after: list[dict]) -> dict:
    """Pair literal checks and degeneration changes by stable case identity."""
    left = {row['id']: row for row in before}
    if set(left) != {row['id'] for row in after}:
        raise ValueError('Paired evaluation populations differ.')
    counts = Counter()
    examples = []
    for row in after:
        original = left[row['id']]
        old, new = (original['metrics'], row['metrics'])
        counts['task_improved'] += not old['task_pass'] and new['task_pass']
        counts['task_regressed'] += old['task_pass'] and (not new['task_pass'])
        counts['degeneration_new'] += not old['severe_degeneration'] and new['severe_degeneration']
        counts['degeneration_fixed'] += old['severe_degeneration'] and (not new['severe_degeneration'])
        counts['lost_END'] += old['ended_with_END'] and (not new['ended_with_END'])
        counts['gained_END'] += not old['ended_with_END'] and new['ended_with_END']
        examples.append(dict(id=row['id'], category=row['category'], prompt=row['prompt'], before=original['response'], after=row['response'], old=old, new=new))
    return dict(counts=dict(counts), cases=examples)

def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError('Four-hour investigation deadline reached.')

def load_model(path: Path) -> LanguageModel:
    payload = read_native_checkpoint(path)
    model = LanguageModel(ModelConfig(**payload['config']['model']))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    if count_parameters(model) != 934356480 or model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError('Architecture/tied embedding changed.')
    return model.to('cuda').eval()

def diagnostics(root: Path, directory: Path, deadline: float) -> dict:
    tokenizer = load_tokenizer(resolve_input(root, 'checkpoints/tokenizer-base-v1/tokenizer.json'))
    completed = root / 'outputs/chat-training/codexa-900m-chat-2h-v1'
    old = root / 'outputs/preparation/codexa-900m-chat-10h-v1'
    initial = json.loads((old / 'deterministic-candidates/codexa-900m-sft-repair-v1/report.json').read_text())
    final = json.loads((completed / 'development/check-0005.json').read_text())
    atomic_document(directory / 'paired-original.json', paired_changes(initial['outputs'], final['outputs']))
    train = IndexedChatDataset(resolve_input(root, 'data/processed/general-chat-sft-v2/train.jsonl'), tokenizer)
    validation = IndexedChatDataset(resolve_input(root, 'data/processed/general-chat-sft-v2/validation.jsonl'), tokenizer)
    rng = random.Random(20261010)
    sampled, summaries = ({}, {})
    for source in ('oasst1', 'ultrachat'):
        items = [i for i, entry in enumerate(validation.index) if entry.source == source]
        rng.shuffle(items)
        sampled[source] = items[:24]
        lengths = [entry.assistant_tokens for entry in train.index if entry.source == source]
        summaries[source] = dict(examples=len(lengths), assistant_tokens=sum(lengths), median_assistant_tokens=statistics.median(lengths), mean_assistant_tokens=statistics.mean(lengths))
    consumed = Counter()
    for line in (completed / 'samples.jsonl').read_text().splitlines():
        consumed.update(json.loads(line)['identities'])
    turns = Counter()
    source_text = defaultdict(list)
    end_count = 0
    with train.path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['conversation_id'] in consumed:
                assistants = [m['content'] for m in row['messages'] if m['role'] == 'assistant']
                turns[row['source']] += len(assistants)
                end_count += len(assistants)
                if len(source_text[row['source']]) < 256:
                    source_text[row['source']].extend(assistants)
    text_metrics = {}
    from codexa_workspace.evaluation import analyze_generated_text
    for source, texts in source_text.items():
        metrics = [analyze_generated_text(text) for text in texts[:256]]
        text_metrics[source] = dict(sampled_responses=len(metrics), severe_text_repetition=sum((m.repeated_ngram_rate > 0.5 for m in metrics)) / len(metrics), mean_repeated_ngram_rate=statistics.mean((m.repeated_ngram_rate for m in metrics)))
    run = json.loads((completed / 'report.json').read_text())
    data_report = dict(source_distribution=summaries, consumed_unique=len(consumed), repeated_conversation_ids=sum((v - 1 for v in consumed.values())), assistant_turns=dict(turns), END_targets=end_count, END_target_fraction=end_count / run['training_state']['tokens_seen'], sampled_training_text_repetition=text_metrics, validation_indices=sampled, limitations=['Source-stratified validation records may have historical SFT exposure.', 'Text-quality sample is first consumed records in physical file order, not unbiased.', 'Conversation identities do not establish unique prefixes or unique linguistic tokens.'])
    atomic_document(directory / 'data-diagnostics.json', data_report)
    index = json.loads((completed / 'checkpoints/recovery-index.json').read_text())
    choices = {'repair': resolve_input(root, 'checkpoints/codexa-900m-sft-repair-v1/latest.pt'), 'balanced': resolve_input(root, 'checkpoints/codexa-900m-sft-balanced-v1/latest.pt'), 'final-2h': completed / 'checkpoints' / index['final']['path']}
    reports = {}
    for name, path in choices.items():
        check_deadline(deadline)
        model = load_model(path)
        source_results = {}
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for source, indices in sampled.items():
                losses, end_losses, early_end_probabilities = ([], [], [])
                for i in indices:
                    check_deadline(deadline)
                    row = validation[i]
                    ids = row['input_ids'].unsqueeze(0).cuda()
                    labels = row['labels'].cuda()
                    logits, _ = model(ids)
                    selected = labels.ne(-100)
                    target = labels[selected]
                    token_logits = logits[0, selected].float()
                    losses.extend(F.cross_entropy(token_logits, target, reduction='none').cpu().tolist())
                    end = target.eq(7)
                    if end.any():
                        end_losses.extend(F.cross_entropy(token_logits[end], target[end], reduction='none').cpu().tolist())
                    if (~end).any():
                        early_end_probabilities.extend(token_logits[~end].softmax(-1)[:, 7].cpu().tolist())
                source_results[source] = dict(assistant_loss=statistics.mean(losses), END_loss=statistics.mean(end_losses), mean_premature_END_probability=statistics.mean(early_end_probabilities), assistant_tokens=len(losses), END_targets=len(end_losses))
        cases = []
        for source, indices in sampled.items():
            for i in indices[:6]:
                entry = validation.index[i]
                with validation.path.open('rb') as stream:
                    stream.seek(entry.offset)
                    row = json.loads(stream.readline())
                history = [ChatMessage(m['role'], m['content']) for m in row['messages'][:-1]]
                if len(tokenizer.encode(format_chat_prompt(history, tokenizer), add_special_tokens=False).ids) > 1536:
                    continue
                cases.append(dict(id=f'local-{source}-{i}', category=source, history=[[m.role, m.content] for m in history[:-1]], prompt=history[-1].content, check='contains', expected=['__not_a_reference_accuracy_test__']))
        local = evaluate_loaded(model, tokenizer, cases, deadline=deadline, output=directory / f'{name}-local-generations.jsonl')
        cache = []
        for case in make_suite()[:4]:
            prompt = tokenizer.encode(format_chat_prompt([ChatMessage('user', case['prompt'])], tokenizer), add_special_tokens=False).ids
            ids = torch.tensor([prompt], device='cuda')
            sequences = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for enabled in (True, False):
                    generated = generate_sequences(model, ids, eos_token_id=2, pad_token_id=0, stop_sequences=[[7]], config=GenerationConfig(max_new_tokens=32, do_sample=False, use_kv_cache=enabled)).sequences[0]
                    sequences.append(list(generated.token_ids))
            cache.append(dict(case_id=case['id'], same_tokens=sequences[0] == sequences[1], cached=sequences[0], uncached=sequences[1]))
        reports[name] = dict(checkpoint=str(path), checkpoint_sha256=file_sha256(path), teacher_forcing=source_results, local_generation=local, cache_parity=cache)
        atomic_document(directory / f'{name}-diagnostics.json', reports[name])
        del model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_document(directory / 'diagnostics.json', dict(data=data_report, models=reports))
    return reports

def pilots(root: Path, directory: Path, deadline: float) -> list[dict]:
    from llm_training.chat_run import run_chat
    planned = [f'{lineage}-{mixture}' for lineage in ('repair', 'balanced') for mixture in ('proportional', 'balanced')]
    pending = sum((not (directory / 'pilots' / name / 'investigation-result.json').exists() for name in planned))
    if shutil.disk_usage(root).free < (100 + pending * 20) * 1024 ** 3:
        raise RuntimeError('Remaining retained pilot recoveries require20GiB each plus100GiB free headroom.')
    parent = json.loads((root / 'outputs/chat-training/codexa-900m-chat-2h-v1/launch.json').read_text())
    tokenizer = load_tokenizer(Path(parent['spec']['tokenizer']['path']))
    results = []
    for lineage in ('repair', 'balanced'):
        for balanced in (False, True):
            check_deadline(deadline)
            name = f"{lineage}-{('balanced' if balanced else 'proportional')}"
            output = directory / 'pilots' / name
            if (output / 'investigation-result.json').exists():
                results.append(json.loads((output / 'investigation-result.json').read_text()))
                continue
            spec = json.loads(json.dumps(parent['spec']))
            checkpoint = resolve_input(root, f'checkpoints/codexa-900m-sft-{lineage}-v1/latest.pt')
            baseline = json.loads((root / f'outputs/preparation/codexa-900m-chat-10h-v1/deterministic-candidates/{checkpoint.parent.name}/report.json').read_text())
            spec['purpose'] = 'investigation'
            spec['initialization'] = dict(path=str(checkpoint), sha256=file_sha256(checkpoint), semantics='weights_initialization')
            spec['training'].update(max_steps=512, warmup_steps=20, checkpoint_interval=256, evaluation_interval=256, seed=73)
            spec['data']['balanced'] = balanced
            spec['baseline_quality'] = baseline['metrics']
            remaining = min(1200, deadline - time.monotonic() - 180)
            if remaining < 300:
                raise TimeoutError('Insufficient time for a bounded pilot and recovery.')
            budget = SessionBudget.start(remaining, 180)
            counter = [0]

            def quality(model, tok, limit):
                counter[0] += 1
                result = evaluate_loaded(model, tok, make_suite(), deadline=limit, output=output / f'quality-{counter[0]}.jsonl')
                atomic_document(output / f'quality-{counter[0]}.json', result)
                return result['metrics']
            report = run_chat(spec, output, budget, quality_evaluator=quality)
            quality_files = sorted(output.glob('quality-*.json'))
            result = dict(name=name, spec=spec, report=report, quality=[json.loads(p.read_text()) for p in quality_files], limitations=['One seed only; equal steps/conversations do not imply equal assistant tokens.', 'All trials use development diagnostics, never the protected acceptance suite.'])
            atomic_document(output / 'investigation-result.json', result)
            results.append(result)
            atomic_document(directory / 'pilot-results.json', results)
            del report
            gc.collect()
            torch.cuda.empty_cache()
    return results

def end_weight_pilot(root: Path, directory: Path, deadline: float) -> dict:
    """Test one predeclared END-weight contrast against repair/proportional training."""
    from llm_training.chat_run import run_chat
    baseline = json.loads((directory/'pilots/repair-proportional/investigation-result.json').read_text())
    spec = json.loads(json.dumps(baseline['spec']))
    spec['optimization']['target_token_weights'] = {'7': 3.0}
    spec['optimization']['objective_description'] = 'sum(assistant CE with END weight3) / actual assistant targets; validation remains unweighted'
    output = directory/'pilots/repair-proportional-END3'
    if (output/'investigation-result.json').exists():
        return json.loads((output/'investigation-result.json').read_text())
    if shutil.disk_usage(root).free < 120 * 1024**3:
        raise RuntimeError('Additional retained pilot requires20GiB plus100GiB headroom.')
    seconds = min(1200, deadline-time.monotonic()-180)
    if seconds < 300:
        raise TimeoutError('Insufficient investigation time for the final bounded contrast.')
    counter = [0]
    def quality(model, tokenizer, limit):
        counter[0] += 1
        result = evaluate_loaded(model, tokenizer, make_suite(), deadline=limit,
                                 output=output/f'quality-{counter[0]}.jsonl')
        atomic_document(output/f'quality-{counter[0]}.json', result)
        return result['metrics']
    report = run_chat(spec, output, SessionBudget.start(seconds, 180), quality_evaluator=quality)
    result = dict(name='repair-proportional-END3', spec=spec, report=report,
                  quality=[json.loads(p.read_text()) for p in sorted(output.glob('quality-*.json'))],
                  limitations=['Same seed/mixture/initialization/update schedule as repair-proportional; END weighting is the sole intended training treatment.',
                               'Weighted training loss cannot be compared directly with unweighted training loss.',
                               'One bounded pilot, not an accepted model or proof of broad capability.'])
    atomic_document(output/'investigation-result.json', result)
    atomic_document(directory/'END-weight-pilot.json', result)
    return result


def cache_diagnostics(root: Path, directory: Path, deadline: float) -> dict:
    """Compare logits on identical prefixes before autoregressive paths can diverge."""
    tokenizer = load_tokenizer(resolve_input(root, 'checkpoints/tokenizer-base-v1/tokenizer.json'))
    model = load_model(resolve_input(root, 'checkpoints/codexa-900m-sft-repair-v1/latest.pt'))
    case = make_suite()[0]
    prompt = tokenizer.encode(format_chat_prompt([ChatMessage('user', case['prompt'])], tokenizer), add_special_tokens=False).ids
    path = json.loads((directory / 'repair-diagnostics.json').read_text())['cache_parity'][0]['cached']
    reports = {}
    try:
        for precision in ('bf16', 'fp32'):
            rows = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                ids = torch.tensor([prompt], device='cuda')
                _, _, cache = model(ids, use_cache=True)
                for position, token in enumerate(path[:-1]):
                    check_deadline(deadline)
                    current = torch.tensor([[token]], device='cuda')
                    cached, _, cache = model(current, past_key_values=cache, use_cache=True)
                    ids = torch.cat((ids, current), dim=1)
                    direct, _ = model(ids)
                    left, right = (cached[0, -1].float(), direct[0, -1].float())
                    cached_top = left.topk(2)
                    direct_top = right.topk(2)
                    rows.append(dict(position=position, prefix_length=ids.shape[1], maximum_absolute_logit_difference=float((left - right).abs().max()), same_argmax=int(left.argmax()) == int(right.argmax()), cached_top_ids=cached_top.indices.cpu().tolist(), direct_top_ids=direct_top.indices.cpu().tolist(), cached_margin=float(cached_top.values[0] - cached_top.values[1]), direct_margin=float(direct_top.values[0] - direct_top.values[1]), cached_END_probability=float(left.softmax(-1)[7]), direct_END_probability=float(right.softmax(-1)[7])))
            reports[precision] = rows
        atomic_document(directory / 'cache-identical-prefix.json', reports)
        return reports
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

def pilot_teacher_forcing(root: Path, directory: Path, deadline: float) -> dict:
    """Compare final ablation checkpoints on the identical source-stratified records."""
    tokenizer = load_tokenizer(resolve_input(root, 'checkpoints/tokenizer-base-v1/tokenizer.json'))
    dataset = IndexedChatDataset(resolve_input(root, 'data/processed/general-chat-sft-v2/validation.jsonl'), tokenizer)
    selected = json.loads((directory / 'data-diagnostics.json').read_text())['validation_indices']
    import numpy as np
    offsets = [0, 100000, 1000000, 10000000, 100000000, 1000000000, 2000000000, 4000000000]
    stream = np.memmap(resolve_input(root, 'data/tokenized/base-v1/mixed/train.bin'), mode='r', dtype=np.uint16)
    windows = np.stack([stream[offset:offset + 513].astype(np.int64) for offset in offsets])
    replay_sha256 = hashlib.sha256(windows.tobytes()).hexdigest()
    results = {}
    trials = json.loads((directory / 'pilot-results.json').read_text())
    if (directory / 'END-weight-pilot.json').exists():
        trials.append(json.loads((directory / 'END-weight-pilot.json').read_text()))
    for trial in trials:
        check_deadline(deadline)
        path = directory / 'pilots' / trial['name'] / 'checkpoints' / trial['report']['recovery_index']['final']['path']
        model = load_model(path)
        report = {}
        try:
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for source, indices in selected.items():
                    assistant, ending = [], []
                    for i in indices:
                        check_deadline(deadline)
                        row = dataset[i]
                        labels = row['labels'].cuda()
                        logits, _ = model(row['input_ids'].unsqueeze(0).cuda())
                        mask = labels.ne(-100)
                        targets = labels[mask]
                        losses = F.cross_entropy(logits[0, mask].float(), targets, reduction='none')
                        assistant.extend(losses.cpu().tolist())
                        ending.extend(losses[targets.eq(7)].cpu().tolist())
                    report[source] = dict(assistant_loss=statistics.mean(assistant), END_loss=statistics.mean(ending),
                                          assistant_tokens=len(assistant), END_targets=len(ending))
            replay_losses = []
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                for window in windows:
                    check_deadline(deadline)
                    _, loss = model(torch.from_numpy(window[:-1].copy()).unsqueeze(0).cuda(),
                                    torch.from_numpy(window[1:].copy()).unsqueeze(0).cuda())
                    replay_losses.append(float(loss))
            results[trial['name']] = dict(checkpoint_sha256=file_sha256(path), sources=report,
                replay_training_loss=statistics.mean(replay_losses), replay_offsets=offsets,
                replay_window_sha256=replay_sha256,
                replay_limitation='Eight base-training windows, not missing base validation or generalization evidence.')
            atomic_document(directory / 'pilot-teacher-forcing.json', results)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return results


def protocol_diagnostics(root: Path, directory: Path) -> dict:
    """Inspect exact inference prefixes and assistant targets on existing records."""
    tokenizer = load_tokenizer(resolve_input(root, 'checkpoints/tokenizer-base-v1/tokenizer.json'))
    dataset = IndexedChatDataset(resolve_input(root, 'data/processed/general-chat-sft-v2/validation.jsonl'), tokenizer)
    selection = json.loads((directory / 'data-diagnostics.json').read_text())['validation_indices']
    checks = []
    for source, indices in selection.items():
        for i in indices:
            with dataset.path.open('rb') as stream:
                stream.seek(dataset.index[i].offset)
                record = json.loads(stream.readline())
            batch = dataset[i]
            ids = batch['input_ids'].tolist()
            labels = batch['labels'].tolist()
            cursor = 0
            history = []
            targets = 0
            turns = 0
            for message in record['messages']:
                content = tokenizer.encode(message['content'], add_special_tokens=False).ids
                segment = [tokenizer.token_to_id('<|' + message['role'] + '|>'), *content, 7]
                assert ids[cursor:cursor + len(segment)] == segment
                if message['role'] == 'assistant':
                    prompt = tokenizer.encode(format_chat_prompt(history, tokenizer), add_special_tokens=False).ids
                    assert prompt == ids[:cursor + 1]
                    turns += 1
                for offset, token in enumerate(segment):
                    position = cursor + offset
                    if position == 0:
                        continue
                    expected = token if message['role'] == 'assistant' and offset > 0 else -100
                    assert labels[position - 1] == expected
                    targets += expected != -100
                history.append(ChatMessage(message['role'], message['content']))
                cursor += len(segment)
            assert labels[-1] == -100 and targets == dataset.index[i].assistant_tokens
            checks.append(dict(index=i, source=source, assistant_turns=turns, supervised_tokens=targets, inference_prefix_equal=True, target_shift_once=True, END_supervised=True))
    report = dict(records=len(checks), checks=checks, passed=True)
    atomic_document(directory / 'protocol-validation.json', report)
    return report

def write_report(root: Path, directory: Path) -> Path:
    """Render actual diagnostic and ablation evidence without promoting a model."""
    diagnostics = json.loads((directory / 'diagnostics.json').read_text())
    paired = json.loads((directory / 'paired-original.json').read_text())
    prefixes = json.loads((directory / 'consumed-prefix-audit.json').read_text())
    trials = json.loads((directory / 'pilot-results.json').read_text())
    if len(trials) != 4:
        raise ValueError('All four predeclared ablations must be accounted for.')
    authorization = json.loads((directory / 'authorization.json').read_text())
    elapsed = time.time() - authorization['started_utc']
    lines = [
        '# CODEXA 934M CONVERSATIONAL REGRESSION INVESTIGATION',
        '',
        'Investigation performed on 10 October 2026 under an explicitly authorized four-hour ceiling.',
        '',
        '## Findings',
        '',
        (
            'No evidence of incorrect role masking, duplicate causal shifting, conversation cycling, or'
            ' tokenizer replacement was found. Existing serializer supervises assistant content and '
            'END, the dataset shifts labels once, and the model compares logits to those shifted '
            'targets without shifting again. Microbatch1 uses complete unpadded conversations.'
        ),
        '',
        f"The completed two-hour experiment contained {prefixes['assistant_turn_instances']:,} assistant turns with only {prefixes['repeated_supervised_prefix_instances']:,} repeated full supervised prefixes. Unique conversation identities do not imply entirely unique context; this audit explicitly accounts for shared prefixes.",
        '',
        f"UltraChat supplied {prefixes['source_supervision']['ultrachat']['assistant_tokens']:,} of 25,912,988 assistant tokens (92.98%). OASST1 supplied {prefixes['source_supervision']['oasst1']['assistant_tokens']:,}. END targets were only {diagnostics['data']['END_target_fraction']:.3%} of supervised tokens. Low prevalence alone does not prove inadequate termination learning.",
        '',
        '## Source-stratified teacher-forcing diagnostic',
        '',
        (
            'Exactly the same 24 validation conversations per source were used for all three '
            'checkpoints. Assistant loss includes END. These records are disjoint from current training'
            ' but may have historical exposure; this is a controlled diagnostic, not proof of '
            'generalization.'
        ),
        '',
        '| Checkpoint | Source | Assistant loss | END loss | Premature END probability |',
        '|---|---|---:|---:|---:|',
    ]
    for name, report in diagnostics['models'].items():
        for source, metrics in report['teacher_forcing'].items():
            lines.append(f"| {name} | {source} | {metrics['assistant_loss']:.5f} | {metrics['END_loss']:.5f} | {metrics['mean_premature_END_probability']:.5f} |")
    lines.extend([
        '',
        (
            'Repair-to-final adaptation improved assistant loss for both sources, but OASST END loss '
            'worsened1.44426→1.89861 while UltraChat END loss improved0.94558→0.77378. This supports '
            'investigating source weighting rather than assuming a global END-mask bug.'
        ),
        '',
        '## Free-running generation and paired changes',
        '',
        (
            'On the same12 local source-stratified conversation prompts, the repair initializer '
            'generated END in 50% and severe degeneration in 41.67%; the two-hour final generated END '
            'in 25% and degeneration in 66.67%. The balanced historical initializer generated END in 25% '
            'and degeneration in 33.33%. These are very small, multilingual diagnostic samples, not a '
            'conversational accuracy benchmark. Dummy local task checks are deliberately ignored.'
        ),
        '',
        f"Across 64 development cases: {paired['counts']}. Literal task macro fell14.29→10.71% through one newly passing case and three newly failing cases. This small change does not establish a statistically reliable general capability regression. Severe repetition remains common regardless of that uncertainty.",
        '',
        (
            'Reference-term scoring can count incoherent answers as correct: the initializer repeatedly'
            ' mentioned Earth in a malformed answer, while the final said only “Yes, humans live on.” '
            'The literal factual failure is real, but neither output is an acceptable answer. No broad '
            'factual-reliability claim follows from keyword scores.'
        ),
        '',
        '## Matched bounded ablations',
        '',
        (
            'Four trials crossed repair/balanced historical initialization with proportional/balanced '
            'source sampling. All used new weights initialization, seed 73, LR5e-6, microbatch1, '
            'accumulation 8, token-normalized loss, complete-conversation dynamic lengths, max512 '
            'updates, warmup20, and a 1,200-second session cap with 180 seconds reserved for recovery. '
            'The visible100×22 Kitty viewer and native recovery store were used. The same development '
            'suite was scored at 256 and 512 updates when reached. No protected acceptance suite was '
            'opened.'
        ),
        '',
        (
            'Equal optimizer steps/conversation count is an operational comparison, not equal '
            'assistant-token exposure: sources have different lengths. One seed and short schedules '
            'cannot establish an optimal mixture.'
        ),
        '',
        (
            '| Trial | Updates | Assistant tokens | OASST examples | UltraChat examples | Final '
            'validation loss | Quality step | Task macro | END | Degeneration | Stop |'
        ),
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
    ])
    eligible = []
    for trial in trials:
        report = trial['report']
        metrics = trial['quality'][-1]['metrics'] if trial['quality'] else {}
        rows = [json.loads(line) for line in (directory / 'pilots' / trial['name'] / 'train_metrics.jsonl').read_text().splitlines()]
        val = [row['validation_loss'] for row in rows if row['validation_loss'] is not None]
        sources = report['counters']['sources']
        lines.append(f"| {trial['name']} | {report['training_state']['optimizer_step']} | {report['training_state']['tokens_seen']:,} | {sources.get('oasst1', 0)} | {sources.get('ultrachat', 0)} | {(val[-1] if val else 'unmeasured')} | {len(trial['quality']) * trial['spec']['training']['evaluation_interval']} | {metrics.get('macro_task_accuracy', 0):.2%} | {metrics.get('END_rate', 0):.2%} | {metrics.get('severe_degeneration_rate', 0):.2%} | {report['stop_reason']} |")
        if metrics.get('entry_gate_passed'):
            eligible.append(trial['name'])
    weighted_path = directory / 'END-weight-pilot.json'
    if weighted_path.exists():
        trial = json.loads(weighted_path.read_text())
        metrics = trial['quality'][-1]['metrics'] if trial['quality'] else {}
        report = trial['report']
        lines.extend(['', '### Additional END-weight contrast', '',
            'This fifth pilot held repair initialization, proportional sampling, seed 73, LR5e-6, accumulation 8 and 512-step schedule fixed. END target 7 received weight 3; other targets remained weight 1. The objective divides the weighted CE sum by actual assistant-token count. Padding and masked roles remain excluded, and evaluation loss remains unweighted. It is a new experimental phase, not a silent objective change in any existing run.', '',
            f"Actual updates {report['training_state']['optimizer_step']}; final recorded development task macro {metrics.get('macro_task_accuracy',0):.2%}, END {metrics.get('END_rate',0):.2%}, degeneration {metrics.get('severe_degeneration_rate',0):.2%}.", '',
            'Raw weighted training loss is not directly comparable with unweighted training loss. The actual token counters are unchanged by weight 3; a CPU engine update matched an independent reference gradient and token count.'])
    teacher_path = directory / 'pilot-teacher-forcing.json'
    if teacher_path.exists():
        lines.extend(['', '### Matched checkpoint END and retention diagnostics', '',
            '| Trial | OASST END loss | UltraChat END loss | Base-training replay loss |',
            '|---|---:|---:|---:|'])
        for name, measured in json.loads(teacher_path.read_text()).items():
            lines.append(f"| {name} | {measured['sources']['oasst1']['END_loss']:.5f} | {measured['sources']['ultrachat']['END_loss']:.5f} | {measured['replay_training_loss']:.5f} |")
        lines.extend(['', 'Replay uses the same eight 512-target windows as the earlier retention diagnostic; these are base-training samples, not replacement validation. Historical repair replay loss was 3.273845 and balanced 3.185360. Window SHA and offsets are retained. No claim of broad retained-capability generalization is made.'])
    lines.extend([
        '',
        f"Entry-gate passing trials: {eligible or 'none'}. The entry floor is a low diagnostic threshold, not promotion approval.",
        '',
        '## Cache investigation',
        '',
    ])
    cache_path = directory / 'cache-identical-prefix.json'
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        for precision, rows in cache.items():
            mismatches = [row for row in rows if not row['same_argmax']]
            lines.append(f"- {precision}: {len(rows)} identical-prefix comparisons; {len(mismatches)} argmax differences; max absolute logit difference {max((row['maximum_absolute_logit_difference'] for row in rows)):.6f}. First differences: {mismatches[:2]}.")
        lines.extend([
            '',
            (
                'The initial cached/uncached greedy test differed on one of four repair prompts after26 '
                'shared tokens. Identical-prefix measurements distinguish numerical sensitivity from an '
                'offset/masking defect; no architecture or cache implementation was changed.'
            ),
        ])
    else:
        lines.append('Identical-prefix follow-up was not completed. Greedy BF16 cache paths differed on one of four repair and final prompts; balanced matched all four. No cache defect has been established.')
    lines.extend([
        '',
        '## Data and evaluation limitations',
        '',
        (
            'A full training-file literal phrase scan found only4 “only the word”,5 “only the number”, '
            'and8 “just the number” occurrences; direct inspection showed the “only the word” hits '
            'mostly describe program output rather than constrain the assistant reply. This is a narrow'
            ' phrase audit, not a semantic inventory. General chat imitation supplies little explicit '
            'practice for this exact evaluation style. Existing repair data has71 training anchors '
            'among9,627 records, with repeated questions/answers and arithmetic targets containing '
            'explanations; this does not teach number-only responses reliably.'
        ),
        '',
        (
            'Historical repair-anchor audit:71 train anchors contain only26 unique conversations;25 '
            'validation anchors contain19 unique conversations, all19 also appear in training. '
            'Arithmetic training anchors have totals42(seven),13(two),17(two). Seven final arithmetic '
            'responses with numerical answers use only those three totals. This is consistent with '
            'narrow template memorization; it is not proof that the exact historical training exposure '
            'alone caused it. Existing General Chat split isolation is a separate issue.'
        ),
        '',
        (
            'No new training examples, datasets, or tokenizer changes were introduced. No END-weighted '
            'objective was added without ablation evidence. All historical assets and the two-hour '
            'final checkpoint remain preserved.'
        ),
        '',
        '## Reproducibility and verification',
        '',
        (
            'Exact protocol inspection passed on 48 real validation conversations: every inference '
            'prefix matches training, every assistant content/END target is shifted once, and '
            'role/user/system targets are ignored. Focused controller/paired-diagnostic tests '
            'passed. Full final test/environment results are recorded in investigation '
            'automated-checks.json when available.'
        ),
        '',
        f'Artifacts: `{directory}`. Current elapsed investigation time at report generation: {elapsed:.1f} seconds of the14,400-second ceiling. Stage start/end records, dataset selections, paired raw outputs, source counts, checkpoint digests, full pilot specifications and native recovery indices are retained.',
        '',
        'Commands from integration root:',
        '',
        '```bash',
        (
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True '
            'OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/investigate_chat_regression.py --stage '
            'diagnostics'
        ),
        (
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True '
            'OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/investigate_chat_regression.py --stage pilots'
        ),
        (
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True '
            'OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/investigate_chat_regression.py --stage cache'
        ),
        (
            'CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTORCH_ALLOC_CONF=expandable_segments:True '
            'OMP_NUM_THREADS=2 .venv/bin/python scripts/legacy/investigate_chat_regression.py --stage report'
        ),
        '```',
        '',
        (
            'Existing pilot result files are reused; completed pilots are never silently trained '
            'longer. Diagnostics create outputs and should be run only in an isolated investigation '
            'directory or with intentional review of existing artifacts.'
        ),
        '',
        '## Decision',
        '',
        (
            'No model is automatically promoted. The completed two-hour final remains technically '
            'recoverable, but a lower loss is insufficient reason to continue its exact schedule. Use '
            'the ablation evidence to choose the next bounded phase, require improvement on both task '
            'behavior and degeneration, and retain the protected final suite for a genuinely final '
            'decision. If no trial passes, continue diagnosis/curation rather than spending the '
            'remaining ceiling on blind repetition.'
        ),
        '',
    ])
    exposure = json.loads((directory / 'historical-exposure-audit.json').read_text())
    role_coverage = json.loads((directory / 'role-and-length-audit.json').read_text())
    lines.extend(['', '## Historical exposure and role coverage', '',
        f"Repair train/validation share {exposure['repair_natural_root_overlap_between_train_and_validation']:,} natural roots and {exposure['repair_exact_content_overlap_between_train_and_validation']} normalized complete-conversation contents. Of the current General Chat validation, {exposure['general_validation_root_overlap_with_repair_train']} records share roots with repair training. None of the96 possibly-clean validation records match repair training by root or normalized complete content; other historical phases remain unproven.", '',
        f"The current training pool contains {role_coverage['message_roles']['assistant']:,} assistant messages and zero system messages. Only {role_coverage['answer_character_counts']['answers_le_32_characters']:,} answers are32 characters or shorter. This is descriptive coverage evidence, not a semantic classification of task quality.", '',
        '## Implemented corrections and ownership', '',
        '- LLM-Data: new data/repair_split.py and revised cli/prepare_repair_sft.py group roots, normalized whole-conversation duplicates and preexisting duplicate components. Stable group hashing makes assignment independent of input order. The existing nonempty-output refusal remains. Only synthetic unit fixtures were split; no historical corpus was regenerated.',
        '- LLM-Training: optional validated quality_guard.py thresholds and patience; token_objective.py target weighting; training.py propagates the optional weighted objective and keeps actual token counts; chat_run.py binds these choices to the saved specification. Defaults preserve existing behavior. New guard policy is proposed for future phases, not retroactively applied to these trials.',
        '- LLM-From-Scratch: investigation driver, script and paired tests; optional behavioral-policy configuration flows through make_spec; proposed configs/chat-behavior-guard-v1.yaml. All relevant wheels rebuilt and installed locally without dependency upgrades. Model/tokenizer/inference architecture remained unchanged.', '',
        'The stricter proposed guard would have flagged two consecutive deteriorating checks by step1600 in the original two-hour run. This is an offline counterfactual, not a new-policy training result. No selected checkpoint or export was promoted.', '',
        '## Test and environment record', ''])
    checks_path = directory / 'automated-checks.json'
    if checks_path.exists():
        for item in json.loads(checks_path.read_text()).get('checks', []):
            lines.append(f"- `{item['command']}`: {item['result']}.")
    lines.extend(['', 'Five integration skips are optional scikit-learn specialist tests in the generative profile. Existing pytest return-value warnings are recorded. Component and integration test counts overlap and must not be summed as unique coverage.', ''])
    path = root / 'documentation/training/CODEXA_900M_CHAT_REGRESSION_INVESTIGATION.md'
    path.write_text('\n'.join(lines))
    return path

def investigate(root: Path, stage: str, *, started: float, seconds: int) -> None:
    directory = root / 'outputs/investigation/chat-regression-20261010'
    directory.mkdir(parents=True, exist_ok=True)
    authorization = directory / 'authorization.json'
    if not authorization.exists():
        atomic_document(authorization, dict(started_utc=time.time() - (time.monotonic() - started), deadline_utc=time.time() - (time.monotonic() - started) + seconds, budget_seconds=seconds, request='User authorized up to four hours investigation with bounded diagnostics/corrective pilots.'))
    record = json.loads(authorization.read_text())
    budget_path = directory / 'investigation-budget.json'
    if not budget_path.exists():
        remaining = record['deadline_utc'] - time.time()
        budget = SessionBudget(record['budget_seconds'], record['started_utc'], record['deadline_utc'],
                               time.monotonic() + remaining, 180)
        atomic_document(budget_path, budget.document())
    if stage == 'report':
        write_report(root, directory)
        return
    budget = SessionBudget.resume(json.loads(budget_path.read_text()))
    deadline = budget.deadline_monotonic
    check_deadline(deadline)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Use deterministic CUBLAS environment.')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    if stage == 'diagnostics':
        diagnostics(root, directory, deadline)
    elif stage == 'pilots':
        pilots(root, directory, deadline)
    elif stage == 'end-weight-pilot':
        end_weight_pilot(root, directory, deadline)
    elif stage == 'cache':
        cache_diagnostics(root, directory, deadline)
    elif stage == 'protocol':
        protocol_diagnostics(root, directory)
    elif stage == 'pilot-teacher':
        pilot_teacher_forcing(root, directory, deadline)
    else:
        write_report(root, directory)
    atomic_document(directory / f'{stage}-completed.json', dict(elapsed_seconds=time.time() - record['started_utc'], completed_utc=time.time()))
    atomic_document(budget_path, budget.document())
