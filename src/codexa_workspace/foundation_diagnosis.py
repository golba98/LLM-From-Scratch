"""Bounded, approval-gated language-versus-chat diagnostics; never training."""

from collections import Counter
from fractions import Fraction
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sqlite3
import statistics
import subprocess
import sys
import time

from llm_architecture.checkpointing import file_sha256
from llm_tokenizer.sft import ChatMessage, format_chat_prompt
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.session import SessionBudget, atomic_document

from .assets import resolve_input
from .chat_evaluation import fingerprint
from .conversational_readiness import resource_preflight, TOKENIZER_SHA256


FORMAT = 'codexa-foundation-diagnostic-v1'
FORMS = ('neutral', 'direct', 'chat', 'short', 'multi_turn')
LIMIT_BYTES = 128 * 1024**2


def concepts() -> list[dict]:
    """Fixed development-only concepts; references are not training examples."""
    rows = [
        ('factual', 'The chemical symbol for carbon is', 'What is the chemical symbol for carbon?', 'C', 'N'),
        ('factual', 'There are this many minutes in one hour:', 'How many minutes are in one hour?', '60', '100'),
        ('factual', 'A hexagon has this many sides:', 'How many sides does a hexagon have?', '6', '8'),
        ('factual', 'A standard chessboard has this many squares:', 'How many squares are on a standard chessboard?', '64', '32'),
        ('factual', 'The capital city of Italy is', 'What is the capital city of Italy?', 'Rome', 'Milan'),
        ('factual', 'The SI unit of electric current is the', 'What is the SI unit of electric current?', 'ampere', 'volt'),
        ('arithmetic', '37 + 26 =', 'What is 37 plus 26?', '63', '53'),
        ('arithmetic', '81 - 34 =', 'What is 81 minus 34?', '47', '57'),
        ('arithmetic', '9 * 8 =', 'What is 9 multiplied by 8?', '72', '56'),
        ('arithmetic', '96 / 8 =', 'What is 96 divided by 8?', '12', '14'),
        ('arithmetic', '18 + 49 =', 'What is 18 plus 49?', '67', '73'),
        ('arithmetic', '72 - 29 =', 'What is 72 minus 29?', '43', '53'),
        ('grammar', 'Yesterday, she [walk/walked] home. The correct choice is', 'Choose walk or walked: Yesterday, she ___ home.', 'walked', 'walk'),
        ('grammar', 'They [is/are] ready. The correct choice is', 'Choose is or are: They ___ ready.', 'are', 'is'),
        ('grammar', 'He owns [a/an] umbrella. The correct choice is', 'Choose a or an: He owns ___ umbrella.', 'an', 'a'),
        ('grammar', 'Two [child/children] arrived. The correct choice is', 'Choose child or children: Two ___ arrived.', 'children', 'child'),
        ('grammar', 'She [has/have] a ticket. The correct choice is', 'Choose has or have: She ___ a ticket.', 'has', 'have'),
        ('grammar', 'This box is [heavy/heavier] than that one. The correct choice is', 'Choose heavy or heavier: This box is ___ than that one.', 'heavier', 'heavy'),
        ('comprehension', 'Lina owns a violet scarf. The colour of her scarf is', 'Lina owns a violet scarf. What colour is her scarf?', 'violet', 'orange'),
        ('comprehension', 'The key was on the desk, then moved into the drawer. It is now in the', 'The key was on the desk, then moved into the drawer. Where is it now?', 'drawer', 'desk'),
        ('comprehension', 'Omar handed his notebook to Priya. The recipient was', 'Omar handed his notebook to Priya. Who received it?', 'Priya', 'Omar'),
        ('comprehension', 'A brass coin is inside a green box. The coin is made of', 'A brass coin is inside a green box. What is the coin made of?', 'brass', 'green'),
        ('comprehension', 'Nora visits before Eli. The earlier visitor is', 'Nora visits before Eli. Who visits earlier?', 'Nora', 'Eli'),
        ('comprehension', 'The meeting moved from Monday to Thursday. Its current day is', 'The meeting moved from Monday to Thursday. On which day is it now?', 'Thursday', 'Monday'),
    ]
    return [dict(id=f'foundation-{i:02}', category=category, prefix=prefix,
                 question=question, expected=answer, distractor=distractor)
            for i, (category, prefix, question, answer, distractor) in enumerate(rows)]


def make_cases(tokenizer) -> list[dict]:
    """Encode five explicitly different formats without injecting BOS."""
    cases = []
    arithmetic_answers = set()
    for concept in concepts():
        if concept['category'] == 'arithmetic':
            match = re.fullmatch(r'(\d+) ([+*/-]) (\d+) =', concept['prefix'])
            if match is None:
                raise ValueError('Unverifiable arithmetic concept.')
            left, operation, right = match.groups()
            left, right = Fraction(left), Fraction(right)
            result = {'+':lambda:left+right, '-':lambda:left-right,
                      '*':lambda:left*right, '/':lambda:left/right}[operation]()
            if str(result) != concept['expected'] or concept['expected'] in arithmetic_answers:
                raise ValueError('Incorrect or repeated arithmetic reference.')
            arithmetic_answers.add(concept['expected'])
        for form in FORMS:
            question = concept['question']
            history = []
            if form == 'neutral':
                prompt = concept['prefix']
            elif form == 'direct':
                prompt = f'Question: {question}\nAnswer:'
            else:
                if form == 'short':
                    question += ' Reply with only the answer, without explanation or punctuation.'
                if form == 'multi_turn':
                    history = [['user', 'We will discuss one question.'],
                               ['assistant', 'Please ask your question.']]
                prompt = format_chat_prompt([ChatMessage(*x) for x in history]
                                            + [ChatMessage('user', question)], tokenizer)
            ids = tokenizer.encode(prompt, add_special_tokens=False).ids
            if not ids or len(ids) + 128 > 2048:
                raise ValueError('Diagnostic prompt is empty or exceeds context.')
            if form in ('neutral', 'direct') and any(value < 8 for value in ids):
                raise ValueError('Plain completion unexpectedly contains special tokens.')
            answer_tokens = {}
            for name in ('expected', 'distractor'):
                # Plain prose needs a separating space. Chat training supervises
                # answer text directly after assistant ID, with no added space.
                text = (' ' if form in ('neutral','direct') else '') + concept[name]
                joint_ids = tokenizer.encode(prompt+text, add_special_tokens=False).ids
                if joint_ids[:len(ids)] != ids:
                    raise ValueError('Answer retokenization changed the conditioning prefix.')
                answer_tokens[name] = joint_ids[len(ids):]
            cases.append(dict(id=f'{concept["id"]}-{form}', concept=concept['id'],
                category=concept['category'], form=form, prompt=prompt, input_ids=ids,
                question=question, history=history, expected=concept['expected'],
                distractor=concept['distractor'], semantic_correct=None,
                answer_token_ids=answer_tokens,
                answer_token_policy='Separating space for plain text; direct assistant text for serialized chat.',
                provenance='locally authored development diagnostic; not historically unseen knowledge',
                format_caveat='multi_turn adds conversational framing, not factual information'))
    return cases


def check_artifact_budget(directory: Path) -> int:
    """Account all owned files; refuse growth beyond the frozen allowance."""
    size = sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    if size > LIMIT_BYTES:
        raise RuntimeError('Diagnostic artifact allowance exceeded.')
    return size


def scoring(text: str, case: dict, terminator: int | None) -> dict:
    """Exact surface answers are distinct from unreviewed semantic judgments."""
    from .evaluation import analyze_generated_text
    exact = text.strip() == case['expected']
    analysis = analyze_generated_text(text).to_dict()
    repeated = analysis['repeated_ngram_rate'] > .5
    return dict(exact_surface_correct=exact, semantic_correct=None,
        arithmetic_exact_correct=exact if case['category'] == 'arithmetic' else None,
        factual_semantic_correct=None, empty=not text.strip(),
        ended_with_END=terminator == 7, ended_with_EOS=terminator == 2,
        chat_turn_stop_expected=case['form'] in ('chat', 'short', 'multi_turn'),
        missing_chat_END=terminator != 7 if case['form'] in ('chat', 'short', 'multi_turn') else None,
        repeated_ngram_severe=repeated, **analysis)


def likelihood_probe(model, tokenizer, case: dict, *, device: str) -> dict:
    """Full answer-token conditional likelihood, not evidence of generated knowledge."""
    import torch
    result = {}
    # Separately appended answer IDs make this conditional token test explicit;
    # it is not silently treated as a retokenized prose continuation.
    for name in ('expected', 'distractor'):
        answer_ids = case['answer_token_ids'][name]
        if not answer_ids:
            raise ValueError('Empty likelihood reference.')
        ids = [*case['input_ids'], *answer_ids]
        logits, _ = model(torch.tensor([ids], device=device, dtype=torch.long))
        if not torch.isfinite(logits).all().item():
            raise FloatingPointError('Nonfinite diagnostic likelihood logits.')
        start = len(case['input_ids']) - 1
        probabilities = logits[0, start:start + len(answer_ids)].float().log_softmax(-1)
        selected = probabilities.gather(-1, torch.tensor(answer_ids, device=device).unsqueeze(-1)).squeeze(-1)
        result[name] = dict(token_ids=answer_ids, mean_token_nll=-selected.mean().item(),
                            first_token_probability=selected[0].exp().item())
    result['limitation'] = 'Token lengths differ; mean NLL is auxiliary, not correctness. Joint encoding preserves each frozen prompt prefix.'
    return result


def evaluate_model(model, tokenizer, cases: list[dict], directory: Path, deadline: float,
                   *, device: str = 'cuda') -> dict:
    """Preserve each completed generation and refuse partial aggregate promotion."""
    import torch
    from contextlib import nullcontext
    from llm_inference.generate import GenerationConfig, generate_sequences
    decode = GenerationConfig(max_new_tokens=128, repetition_penalty=1, seed=42, do_sample=False)
    output = directory/'outputs.jsonl'
    rows = []
    model.eval()
    context = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if device == 'cuda' else nullcontext()
    def finite_decode_logits(_model, _inputs, output):
        if not torch.isfinite(output[0][:, -1, :]).all().item():
            raise FloatingPointError('Nonfinite free-generation logits; no answer is scored.')
    # The existing greedy decoder does not reject NaN/Inf before argmax. Observe
    # every decode forward without changing generation mathematics or cache use.
    with torch.inference_mode(), context, model.register_forward_hook(finite_decode_logits):
        for case in cases:
            if time.monotonic() >= deadline:
                raise TimeoutError('Original diagnostic deadline expired.')
            inputs = torch.tensor([case['input_ids']], device=device, dtype=torch.long)
            generated = generate_sequences(model, inputs, eos_token_id=2, pad_token_id=0,
                stop_sequences=[[7]], config=decode).sequences[0]
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline expired during generation.')
            text = tokenizer.decode(list(generated.visible_token_ids), skip_special_tokens=True).strip()
            row = dict(**case, response=text, raw_token_ids=list(generated.token_ids),
                terminator=generated.terminating_token_id, termination=generated.termination_cause,
                generated_tokens=generated.generated_token_count,
                unexpected_role_token_count=sum(token in (4,5,6) for token in generated.token_ids),
                metrics=scoring(text, case, generated.terminating_token_id),
                likelihood=likelihood_probe(model, tokenizer, case, device=device))
            if time.monotonic() >= deadline:
                raise TimeoutError('Diagnostic deadline expired during likelihood probes.')
            with output.open('a') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            rows.append(row)
            check_artifact_budget(directory.parent)
    return dict(cases=len(rows), outputs_path=str(output), semantic_review_pending=len(rows),
        exact_surface_by_form={form: sum(x['metrics']['exact_surface_correct'] for x in rows
            if x['form'] == form) / sum(x['form'] == form for x in rows) for form in FORMS},
        completed=True, training_updates=0)


def prepare_blinded_review(directory: Path, results: dict) -> dict:
    """Use the established review page; model names are absent from reviewer inputs."""
    from .qualification_review import write_review_page
    names = sorted(results)
    random.Random(42).shuffle(names)
    mapping = {f'Model-{chr(65+i)}': name for i,name in enumerate(names)}
    items = []
    for alias,name in mapping.items():
        outputs = [json.loads(line) for line in Path(results[name]['outputs_path']).open()]
        for row in outputs:
            items.append(dict(id=f'{alias}:{row["id"]}', kind='model',
                group=f'{row["category"]} / {row["form"]}', instruction=row['prompt'],
                answer=row['response'], provenance=dict(candidate_alias=alias, case=row['id']),
                findings=dict(reference=row['expected'], metrics=row['metrics']),
                warnings=['Approve only a correct complete response without contradictory additions.',
                          'Reject incorrect or unsupported responses. Leave uncertain semantics pending.']))
    review = directory/'review'
    atomic_document(directory/'private-review-mapping.json', mapping)
    package = write_review_page(review, dict(format='codexa-foundation-review-package-v1',
        items=items, candidate_aliases=list(mapping),
        mapping_sha256=fingerprint(mapping), output_identity=fingerprint(items),
        rubric='Decision approves semantic correctness of the entire response, not keyword presence.'))
    return package


def reviewed_comparison(directory: Path, decisions: Path) -> dict:
    """Summarize genuine complete decisions; keep semantic unknowns out of scores."""
    package = json.loads((directory/'review/package.json').read_text())
    mapping = json.loads((directory/'private-review-mapping.json').read_text())
    export = json.loads(decisions.read_text())
    if (export.get('format') != 'codexa-human-review-export-v1'
            or export.get('package_sha256') != package['package_sha256']
            or not str(export.get('reviewer', '')).strip()
            or export.get('method') != 'human_manual_review'):
        raise ValueError('A genuine fingerprint-bound human review export is required.')
    if fingerprint(mapping) != package['mapping_sha256']:
        raise ValueError('Blinded mapping drift.')
    # Rebuild the package fingerprint independently, not just its embedded field.
    if fingerprint({k:v for k,v in package.items() if k != 'package_sha256'}) != package['package_sha256']:
        raise ValueError('Review package drift.')
    ids = {x['id'] for x in package['items']}
    actual = export.get('decisions', {})
    if set(actual) != ids or any(actual[key].get('decision') not in ('approve','reject') for key in ids):
        raise ValueError('Every diagnostic response needs an explicit semantic decision.')
    comparison = json.loads((directory/'comparison.json').read_text())
    # Validate the current raw outputs against the exact blinded package.
    package_rows = {x['id']:x for x in package['items']}
    by_name = {}
    for alias,name in mapping.items():
        rows = [json.loads(line) for line in Path(comparison['results'][name]['outputs_path']).open()]
        if len(rows) != 120:
            raise ValueError('Incomplete generation population.')
        scores = {}
        for row in rows:
            key = f'{alias}:{row["id"]}'
            item = package_rows[key]
            if (row['response'] != item['answer'] or row['prompt'] != item['instruction']
                    or row['expected'] != item['findings']['reference']
                    or row['metrics'] != item['findings']['metrics']):
                raise ValueError('Raw diagnostic outputs changed after review publication.')
            scores[(row['concept'],row['form'])] = actual[key]['decision'] == 'approve'
        by_name[name] = scores
    rng = random.Random(42)
    concept_ids = [x['id'] for x in concepts()]
    paired = []
    eligible = []
    for name,scores in by_name.items():
        for form in ('neutral','direct'):
            correct = sum(scores[(key,form)] for key in concept_ids)
            factual = sum(scores[(x['id'],form)] for x in concepts() if x['category']=='factual')
            comprehension = sum(scores[(x['id'],form)] for x in concepts() if x['category']=='comprehension')
            differences = [int(scores[(key,form)])-int(scores[(key,'chat')]) for key in concept_ids]
            estimates = sorted(sum(rng.choice(differences) for _ in concept_ids)/24 for _ in range(2000))
            # Ten comparisons (five models, two plain forms): Bonferroni 99.5% intervals.
            lower,upper = estimates[5], estimates[1994]
            pair = dict(candidate=name, form=form, semantic_correct=correct, factual_correct=factual,
                comprehension_correct=comprehension, paired_gain_cases=sum(differences),
                family_cluster_interval_99_5=[lower,upper], bootstrap_samples=2000,
                format_sensitive=sum(differences)>=6 and lower>0)
            paired.append(pair)
            if correct>=12 and factual>=4 and comprehension>=4 and pair['format_sensitive']:
                eligible.append(dict(candidate=name, form=form))
    return dict(reviewer=export['reviewer'], decisions_sha256=file_sha256(decisions),
        reviewed_responses=len(ids), paired=paired, corrective_sft_diagnostic_signals=eligible,
        initializer_selected=False, training_authorized=False,
        limitation='Small development diagnostic; historical and semantic exposure not excluded.')


def code_identity(root: Path) -> dict:
    """Bind the orchestrator and inference implementation to authorization."""
    import llm_inference.generate as generation
    import llm_tokenizer.sft as serializer
    import llm_architecture.model as architecture
    import llm_architecture.checkpointing as checkpoint_reader
    from . import chat_operations
    paths = [Path(__file__), root/'scripts/conversational_readiness.py',
             Path(generation.__file__), Path(serializer.__file__), Path(chat_operations.__file__),
             Path(architecture.__file__), Path(checkpoint_reader.__file__),
             Path(__file__).with_name('qualification_review.py'),
             Path(__file__).with_name('conversational_readiness.py')]
    return {str(path): file_sha256(path) for path in paths}


def build(root: Path, directory: Path) -> dict:
    """Prepare small immutable job inputs using existing candidate identities."""
    if directory.exists():
        raise FileExistsError('Use a fresh diagnostic namespace.')
    from .assets import output_root
    if not directory.resolve().is_relative_to(output_root(root).resolve()):
        raise ValueError('Diagnostic must use the configured output root.')
    prior = root/'outputs/qualification/codexa-934m-20261011-v1/baseline-generations/comparison.json'
    previous = json.loads(prior.read_text())
    tokenizer_path = resolve_input(root, 'checkpoints/tokenizer-base-v1/tokenizer.json')
    if file_sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError('Tokenizer changed.')
    cases = make_cases(load_tokenizer(tokenizer_path))
    # Existing results are read-only evidence; no protected acceptance answers.
    prompts = {x['prompt'].strip().casefold() for v in previous['results'].values() for x in v['outputs']}
    overlaps = [x['id'] for x in cases if x['question'].strip().casefold() in prompts]
    if overlaps:
        raise ValueError('Exact existing development prompt overlap: ' + str(overlaps))
    from llm_data.conversational_audit import content_identity
    identities = {content_identity([*[dict(role=role, content=text) for role,text in case['history']],
        dict(role='user', content=case['question']), dict(role='assistant', content=case['expected'])])
        for case in cases}
    ledger = root/'outputs/readiness/codexa-934m-20261010-v1/conversation-audit.sqlite'
    connection = sqlite3.connect(f'file:{ledger.absolute()}?mode=ro', uri=True)
    try:
        matches = [dict(dataset=dataset, split=split, content=value)
                   for dataset,split,value in connection.execute('SELECT dataset,split,content FROM records')
                   if value in identities]
    finally:
        connection.close()
    if matches:
        raise ValueError('Exact historical conversation overlap: '+str(matches))
    plan = dict(format=FORMAT, candidates=previous['plan']['candidates'],
        candidate_order=['base','repair','balanced','original_2h','END3'],
        tokenizer_path=str(tokenizer_path), tokenizer_sha256=TOKENIZER_SHA256,
        suite_sha256=fingerprint(cases), concepts=24, forms=list(FORMS),
        cases_per_candidate=120, candidates_total=5, total_generations=600,
        decoding=dict(max_new_tokens=128, do_sample=False, repetition_penalty=1, seed=42),
        numerics=dict(autocast='bf16', weights='fp32', deterministic_algorithms=False,
                      cublas_workspace_config=':4096:8',
                      limitation='Matches prior inference settings; not a bitwise CUDA determinism claim.'),
        max_wall_clock_seconds=3600, optimizer_updates=0, artifact_limit_bytes=LIMIT_BYTES,
        code_identity=code_identity(root), source_comparison_sha256=file_sha256(prior),
        output=str(directory), approval_format='codexa-foundation-diagnostic-approval-v1',
        thresholds=dict(format_gain_cases=6, neutral_direct_correct=12,
            factual_correct=4, comprehension_correct=4, weak_overall_max=6,
            weak_factual_max=2, weak_comprehension_max=2,
            bootstrap_samples=2000, pairwise_interval_confidence=.995),
        protected_acceptance_used=False, human_review_required=True)
    plan['contamination_checks'] = dict(exact_existing_prompt_overlaps=overlaps,
        exact_historical_conversation_matches=matches,
        semantic_and_template_overlap='not excluded; common facts and skills are deliberately shared')
    plan['job_sha256'] = fingerprint(plan)
    directory.mkdir(parents=True)
    atomic_document(directory/'suite.json', cases)
    atomic_document(directory/'plan.json', plan)
    atomic_document(directory/'resources.json', resource_preflight(root))
    atomic_document(directory/'approval-request.json', dict(format=plan['approval_format'],
        job_sha256=plan['job_sha256'], approved=False, purpose='inference only',
        max_wall_clock_seconds=3600, optimizer_updates=0,
        required_action='Provide genuine job-specific approval; this request is not authorization.'))
    check_artifact_budget(directory)
    return plan


def validate_job(root: Path, directory: Path, approval: Path, deadline: float) -> dict:
    """Reject stale code, suites, approval, expired sessions and live resource failure."""
    if Path(sys.prefix).resolve() != (root/'.venv').resolve():
        raise RuntimeError('Use the integration generative environment.')
    plan = json.loads((directory/'plan.json').read_text())
    if fingerprint({k:v for k,v in plan.items() if k != 'job_sha256'}) != plan['job_sha256']:
        raise ValueError('Diagnostic plan drift.')
    if plan['format'] != FORMAT or plan['optimizer_updates'] != 0:
        raise ValueError('Unknown diagnostic or training requested.')
    if fingerprint(json.loads((directory/'suite.json').read_text())) != plan['suite_sha256']:
        raise ValueError('Diagnostic suite drift.')
    if code_identity(root) != plan['code_identity']:
        raise ValueError('Approved implementation drift.')
    grant = json.loads(approval.read_text())
    if (grant.get('format') != plan['approval_format'] or grant.get('approved') is not True
            or grant.get('job_sha256') != plan['job_sha256']):
        raise ValueError('Precise diagnostic approval is missing or mismatched.')
    if not math.isfinite(deadline) or time.monotonic() >= deadline:
        raise TimeoutError('Original diagnostic deadline expired.')
    if plan.get('numerics') and os.environ.get('CUBLAS_WORKSPACE_CONFIG') != plan['numerics']['cublas_workspace_config']:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must match the frozen prior inference protocol.')
    resources = resource_preflight(root)
    if not all(resources[k] for k in ('disk_gate_passed', 'host_gate_passed', 'gpu_gate_passed')):
        raise RuntimeError('Mandatory resource gate failed; no model loaded.')
    if file_sha256(Path(plan['tokenizer_path'])) != plan['tokenizer_sha256']:
        raise ValueError('Tokenizer identity drift.')
    check_artifact_budget(directory)
    return plan


def candidate(root: Path, directory: Path, approval: Path, name: str, deadline: float) -> dict:
    """One process owns one checkpoint and exits; original deadline is enforced."""
    import torch
    from llm_architecture.checkpointing import read_native_checkpoint
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
    plan = validate_job(root, directory, approval, deadline)
    recorded = json.loads((directory/'supervisor/budget.json').read_text())
    if deadline != recorded['deadline_monotonic']:
        raise ValueError('Child cannot reset the recorded deadline.')
    record = plan['candidates'][name]
    checkpoint = Path(record['path'])
    if file_sha256(checkpoint) != record['sha256']:
        raise ValueError('Checkpoint identity changed before load.')
    target = directory/name
    target.mkdir(exist_ok=False)
    started = time.monotonic()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(plan['numerics']['deterministic_algorithms'])
    payload = read_native_checkpoint(checkpoint)
    model = LanguageModel(ModelConfig(**payload['config']['model']))
    model.load_state_dict(payload['model_state_dict'], strict=True)
    del payload
    model.to('cuda').eval()
    if count_parameters(model) != 934356480 or model.lm_head.weight is not model.token_embeddings.weight:
        raise ValueError('Architecture compatibility failed.')
    try:
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all().item():
                raise FloatingPointError('Nonfinite weights.')
        tokenizer = load_tokenizer(Path(plan['tokenizer_path']))
        result = evaluate_model(model, tokenizer, json.loads((directory/'suite.json').read_text()),
                                target, deadline)
        if file_sha256(checkpoint) != record['sha256']:
            raise ValueError('Checkpoint changed during evaluation.')
        result.update(checkpoint=record, seconds=time.monotonic()-started,
                      numerics=dict(plan['numerics'], allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        atomic_document(target/'report.json', result)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def run(root: Path, directory: Path, approval: Path, started: float) -> dict:
    """Reuse owned-process supervision with one nonrenewable 3,600-second session."""
    deadline = started + 3600
    plan = validate_job(root, directory, approval, deadline)
    from .chat_operations import watch_owned_process
    control = directory/'supervisor'
    control.mkdir(exist_ok=False)
    budget = SessionBudget.start(3600, 30)
    budget.deadline_monotonic = deadline
    budget.started_utc = time.time() - (time.monotonic()-started)
    budget.deadline_utc = budget.started_utc + 3600
    atomic_document(control/'budget.json', budget.document())
    results = {}
    try:
        for name in plan.get('candidate_order', list(plan['candidates'])):
            validate_job(root, directory, approval, deadline)
            child_control = control/name
            child_control.mkdir()
            command = [sys.executable, str(root/'scripts/conversational_readiness.py'),
                'foundation-candidate', '--output', str(directory), '--approval-file', str(approval),
                '--candidate', name, '--original-started-monotonic', str(started)]
            with (child_control/'console.log').open('x') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True, cwd=root)
                atomic_document(child_control/'process.json', dict(pid=child.pid, deadline=deadline))
                try:
                    code = watch_owned_process(child, budget, child_control)
                finally:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=max(.1, min(5, budget.remaining)))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
            if code:
                raise RuntimeError(f'Candidate {name} failed with status {code}.')
            result = json.loads((directory/name/'report.json').read_text())
            if result['cases'] != 120 or not result['completed']:
                raise ValueError('Incomplete candidate cannot enter comparison.')
            results[name] = result
            atomic_document(directory/'completed-evidence.json', results)
        atomic_document(directory/'comparison.json', dict(plan=plan, results=results,
            semantic_review_complete=False, training_decision='INSUFFICIENT EVIDENCE — TARGETED DIAGNOSTIC REQUIRED'))
        prepare_blinded_review(directory, results)
        check_artifact_budget(directory)
        return results
    except BaseException as error:
        atomic_document(directory/'incomplete.json', dict(error=f'{type(error).__name__}: {error}',
            completed_candidates=list(results), budget=budget.document(), training_updates=0))
        raise


def training_history(asset_root: Path) -> dict:
    """Summarize original metrics without conflating unrelated later run lineages."""
    log = asset_root/'logs/codexa-900m-base-v1'
    metadata = json.loads((log/'run_metadata.json').read_text())
    rows = [json.loads(line) for line in (log/'train_metrics.jsonl').open()]
    windows = []
    for index in range(0, len(rows), 250):
        block = rows[index:index+250]
        windows.append(dict(first_step=block[0]['optimizer_step'], last_step=block[-1]['optimizer_step'],
            training_loss_mean=statistics.mean(x['training_loss'] for x in block),
            learning_rate=block[-1]['learning_rate'], validation_loss=block[-1]['validation_loss'],
            median_tokens_per_second=statistics.median(x['tokens_per_second'] for x in block),
            gradient_norm_mean=statistics.mean(x['gradient_norm'] for x in block)))
    manifest_path = asset_root/'data/tokenized/base-v1/mixed/token_data_manifest.json'
    corpus = json.loads(manifest_path.read_text())
    return dict(selected_run=metadata, windows=windows, corpus=corpus,
        processed_tokens_per_parameter=metadata['tokens_seen']/metadata['total_parameter_count'],
        processed_fraction_of_corpus=metadata['tokens_seen']/corpus['train_token_count'],
        historical_validation_available=(asset_root/'data/tokenized/base-v1/fineweb_edu/validation.bin').exists(),
        repeated_training_windows='not established from metric totals alone',
        source_hashes={str(x):file_sha256(x) for x in [log/'run_metadata.json', log/'train_metrics.jsonl', manifest_path]})


def corpus_sample(asset_root: Path, tokenizer) -> dict:
    """Read 128 documents/source from a bounded mixed-index prefix, not whole corpus."""
    import numpy as np
    index = asset_root/'data/tokenized/base-v1/mixed/train_index.json'
    token_file = asset_root/'data/tokenized/base-v1/mixed/train.bin'
    tokens = np.memmap(token_file, dtype=np.uint16, mode='r')
    counts = Counter()
    rows = []
    scanned = 0
    with index.open() as stream:
        for line in stream:
            scanned += len(line.encode())
            if scanned > 4 * 1024**2:
                break
            stripped = line.strip().rstrip(',')
            if not stripped or stripped in ('[', ']'):
                continue
            record = json.loads(stripped)
            source = record['source']
            if source not in ('fineweb_edu', 'wikipedia') or counts[source] >= 128:
                continue
            start, end = record['token_start'], record['token_end']
            end_sample = min(end, start+16384)
            ids = [int(x) for x in tokens[start:end_sample]]
            text = tokenizer.decode(ids, skip_special_tokens=False)
            restored = tokenizer.encode(text, add_special_tokens=False).ids
            visible = tokenizer.decode(ids, skip_special_tokens=True)
            words = visible.split()
            lexical_words = re.findall(r'[a-z]+', visible.casefold())
            english_function_words = {'the','a','an','and','of','to','in','is','was','are','were',
                'for','with','that','this','on','by','as','it','from','be','has','have','not','or'}
            function_fraction = sum(x in english_function_words for x in lexical_words)/max(1,len(lexical_words))
            nonempty_lines = [x.strip() for x in visible.splitlines() if x.strip()]
            rows.append(dict(source=source, document_id=record['document_id'],
                ordinal=record['document_ordinal'], token_start=start, token_end=end,
                sampled_tokens=len(ids), truncated=end_sample < end,
                utf8_bytes=len(visible.encode()), words=len(words),
                tokens_per_word=len(ids)/max(1,len(words)), byte_roundtrip_ids_equal=restored == ids,
                replacement_characters=visible.count('\ufffd'),
                ascii_letter_fraction=sum('a' <= x.lower() <= 'z' for x in visible)/max(1,len(visible)),
                english_function_word_fraction=function_fraction,
                language_screen='english_like' if len(lexical_words)>=20 and function_fraction>=.05 else 'unknown',
                repeated_nonempty_line_fraction=(len(nonempty_lines)-len(set(nonempty_lines)))/max(1,len(nonempty_lines)),
                language_verified=None, duplicate_text_sha256=hashlib.sha256(visible.encode()).hexdigest(),
                excerpt=visible[:1200]))
            counts[source] += 1
            if all(counts[source] == 128 for source in ('fineweb_edu', 'wikipedia')):
                break
    del tokens
    return dict(method='first 128 per source in shuffled mixed-index prefix; not corpus-wide random sampling',
        index_bytes_read=scanned, counts=dict(counts), documents=rows,
        duplicate_sample_texts=len(rows)-len({x['duplicate_text_sha256'] for x in rows}),
        language_limitation='ASCII proportion is not language identification; human language review pending',
        max_document_sample_tokens=16384)
