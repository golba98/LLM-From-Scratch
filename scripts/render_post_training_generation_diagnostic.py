"""Build one organized report from measured diagnostic evidence; no model execution."""
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
import statistics
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
D: Path
RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1"


def read(name):
    return json.loads((D/name).read_text())


def rows(name):
    return [json.loads(s) for s in (D/name).read_text().splitlines()]


def output(name, text):
    (D/name).write_text(text.strip()+"\n")


def source(repo, name, line):
    path = ROOT / name if repo == "integration" else ROOT.parent / repo / name
    return f"[{repo}: {Path(name).name}:{line}](<{path}:{line}>)"


def excerpt(text, limit=480):
    return "```text\n"+text[:limit]+"\n```\n"+(f"*Excerpt: first {limit} characters; complete raw text is in the JSONL.*\n" if len(text)>limit else "*Complete raw generation.*\n")


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    plot_training_curves()
    manifest=read('manifest.json');dataset=read('dataset.json');target_audit=read('full-selected-target-audit.json')
    training=read('training-analysis.json');baseline=read('checkpoint-baseline.json');final=read('checkpoint-final.json')
    historical=read('historical-summary.json');validation={x:read(f'validation-{x}.json') for x in ('baseline','final')}
    generations=rows('generations-baseline.jsonl')+rows('generations-final.jsonl')
    assert len(generations)==400 and len({(x['checkpoint'],x['suite'],x['decode'],x['id']) for x in generations})==400
    groups=defaultdict(list)
    for row in generations:groups[(row['checkpoint'],row['suite'],row['decode'])].append(row)
    metrics=[]
    for key,group in groups.items():
        metrics.append(dict(checkpoint=key[0],suite=key[1],decode=key[2],cases=len(group),
            exact_surface_matches=sum(x.get('exact_surface_match',False) for x in group),
            exact_scored_cases=sum(x.get('expected') is not None for x in group),
            historical_severe=sum(x['metrics']['historical_severe'] for x in group),
            independent_severe=sum(x['metrics']['independent_severe'] for x in group),
            length=sum(x['finish_reason']=='length' for x in group),EOS=sum(x['terminator']==2 for x in group),END=sum(x['terminator']==7 for x in group),
            mean_token_distinct_1=statistics.mean(x['metrics']['token_distinct_1'] for x in group),
            mean_token_distinct_2=statistics.mean(x['metrics']['token_distinct_2'] for x in group),
            mean_generated_tokens=statistics.mean(x['generated_tokens'] for x in group),
            historical_token_matches=sum(x.get('historical_token_match',False) for x in group)))
    output('generation-summary.json',json.dumps(metrics,indent=2))
    paired_index={(x['checkpoint'],x['id']):x for x in generations if x['suite']=='historical' and x['decode']=='greedy'}
    paired=[]
    for case_id in sorted(x['id'] for x in generations if x['checkpoint']=='baseline' and x['suite']=='historical' and x['decode']=='greedy'):
        a,b=paired_index[('baseline',case_id)],paired_index[('final',case_id)]
        paired.append(dict(id=case_id,identical_token_arrays=a['raw_token_ids']==b['raw_token_ids'],
            first_token_changed=a['raw_token_ids'][0]!=b['raw_token_ids'][0],
            baseline_repetition=a['metrics']['independent_severe'],final_repetition=b['metrics']['independent_severe'],
            baseline_termination=a['termination'],final_termination=b['termination']))
    identical_pairs=sum(x['identical_token_arrays'] for x in paired)
    output('baseline-final-paired.json',json.dumps(paired,indent=2))
    with (D/'generation-summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    with (D/'generation-metrics.csv').open('w') as stream:
        flattened=[dict(checkpoint=x['checkpoint'],suite=x['suite'],decode=x['decode'],id=x['id'],
            generated_tokens=x['generated_tokens'],termination=x['termination'],exact_surface_match=x.get('exact_surface_match'),**x['metrics']) for x in generations]
        writer=csv.DictWriter(stream,fieldnames=list(flattened[0]));writer.writeheader();writer.writerows(flattened)
    comparisons=read('cache-equivalence.json');long_checks=read('cache-boundary-fp32-followup.json');precision=read('precision-localization.json')
    masked=read('masked-cache-isolated-correction.json');cache_generation=rows('free-generation-cache-comparison.jsonl');traces=rows('token-traces.jsonl')
    doc_probes={x:rows(f'document-eos-probes-{x}.jsonl') for x in ('baseline','final')}
    runtime_files=[p.name for p in D.glob('checkpoint-*.json') if 'runtime' in read(p.name)]
    peak_allocated=max(read(x)['runtime']['peak_allocated_bytes'] for x in runtime_files)
    peak_reserved=max(read(x)['runtime']['peak_reserved_bytes'] for x in runtime_files)
    # Verify all original critical inputs again, not just the final model.
    critical=[Path(final['path']),Path(baseline['path']),Path(json.loads((RUN/'config/job.json').read_text())['spec']['tokenizer']['path'])]
    preservation=[]
    for path in critical:
        actual=digest(path);expected=manifest['files'][str(path)]['sha256']
        preservation.append(dict(path=str(path),before=expected,after=actual,unchanged=actual==expected))
    assert all(x['unchanged'] for x in preservation)
    output('preservation-final.json',json.dumps(preservation,indent=2))
    assert validation['baseline']['weighted_loss']==json.loads((RUN/'metrics/report.json').read_text())['counters']['baseline_validation_loss']
    assert validation['final']['weighted_loss']==json.loads((RUN/'metrics/report.json').read_text())['counters']['best_validation_loss']
    expected_source=source('LLM-Architecture','src/llm_architecture/model.py',479)
    generator_source=source('LLM-Inference','src/llm_inference/generate.py',336)
    mixture_source=source('integration','src/codexa_workspace/foundation_training.py',90)
    historical_source=source('integration','src/codexa_workspace/foundation_training.py',365)
    target_source=source('LLM-Data','src/llm_data/token_data.py',573)
    loss_source=source('LLM-Architecture','src/llm_architecture/model.py',532)
    eos_min=min(x['expected_answer_end']['eos_probability'] for x in traces)
    eos_max=max(x['expected_answer_end']['eos_probability'] for x in traces)
    end_max=max(x['expected_answer_end']['end_probability'] for x in traces)
    eos_table='| Case | EOS rank | EOS probability | END rank | END probability |\n|---|---:|---:|---:|---:|\n'
    eos_table+='\n'.join(f"| {x['id']} | {x['expected_answer_end']['eos_rank']} | {x['expected_answer_end']['eos_probability']:.7g} | {x['expected_answer_end']['end_rank']} | {x['expected_answer_end']['end_probability']:.7g} |" for x in traces)
    common=f"Evidence directory: `{D}`. Reproduction uses the existing `.venv`, no dependency installation. All human ratings remain pending."
    reproduction=f"""From the integration repository, choose a fresh output directory and run:

```sh
.venv/bin/python scripts/post_training_generation_diagnostic.py --phase prepare --directory outputs/diagnostics/post-training-generation-investigation/NEW-ID
.venv/bin/python scripts/post_training_generation_diagnostic.py --phase baseline --directory outputs/diagnostics/post-training-generation-investigation/NEW-ID
.venv/bin/python scripts/post_training_generation_diagnostic.py --phase final --directory outputs/diagnostics/post-training-generation-investigation/NEW-ID
.venv/bin/python scripts/render_post_training_generation_diagnostic.py --directory outputs/diagnostics/post-training-generation-investigation/NEW-ID
.venv/bin/python -m pytest -q -p no:cacheprovider tests/test_post_training_generation_diagnostic.py
```

Each phase has a 30-minute alarm; completed accelerated attempts share a 90-minute ceiling. GPU allocation is limited to half the 16 GiB card, with an 8 GiB allocation ceiling and 2 GiB physical-headroom guard. `--attempt N` records a separate diagnostic attempt and reuses complete generation rows only; it never resumes training. Historical runs cannot be used as output directories.
"""
    findings=[
        dict(id='objective-mismatch',classification='CONFIRMED',confidence='high',symptom='No useful conversational behavior despite improving general-text CE',
            evidence='Complete source-index accounting: 71,696,026 FineWeb-Edu and 45,744,486 Wikipedia targets; all 117,440,512 targets scanned: zero role/END targets. Current objective is unmasked next-token pretraining, not assistant-only SFT.',
            sources=[mixture_source,target_source,loss_source],reproduction='prepare phase; dataset.json and full-selected-target-audit.json',
            effect='Optimizes document continuation, with no positive chat-turn END supervision in this run. Does not by itself explain every plain-completion failure.',
            fix='A small, reviewed, complete-turn assistant-only SFT feasibility experiment; maintain exact chat-v1 serialization.',
            risks='Historical SFT improved stopping without reliably improving usefulness; formatting success must not count as conversational success.'),
        dict(id='greedy-degeneration',classification='CONFIRMED',confidence='high',symptom='46/48 severe greedy repetitions; no exact generation improvements',
            evidence='96/96 historical greedy outputs reproduced exactly. Final temperature-1 sampling: 0/48 severe; nucleus: 15/48 independent severe; nucleus plus penalty 1.1: 0/48 severe. All retain 0/48 exact surface matches.',
            sources=[generator_source,historical_source],reproduction='final decoding matrix; generation-summary.csv and generations-final.jsonl',
            effect='Greedy mode repeatedly chooses its own high-probability continuation; bounded decoding changes can reduce mechanical loops but do not recover demonstrated task competence. Baseline-to-final text changes on 35/48 cases while failure aggregates remain unchanged.',
            fix='Use sampling as a diagnostic control while correcting the learning objective; do not present repetition suppression as model repair.',
            risks='Sampling makes fluent-looking hallucinations, and exact matching is narrower than semantic correctness. No semantic scores have been invented.'),
        dict(id='response-termination-not-learned',classification='CONFIRMED',confidence='high for audited response positions',symptom='47/48 historical length endings; no END in any of 400 new generations',
            evidence=f'Forced EOS/END work on CPU/CUDA. Eight expected-answer endings: EOS probability {eos_min:.6g}–{eos_max:.6g}; END at most {end_max:.6g}. Actual document endings differ: final mean EOS probability {statistics.mean(x["distribution"]["eos_probability"] for x in doc_probes["final"]):.4f}, rank 1 on 5/32.',
            sources=[generator_source,source('LLM-Tokenizer','src/llm_tokenizer/sft.py',133)],
            reproduction='token-traces.jsonl, document-eos-probes-*.jsonl, forced-stop tests',
            effect='The engine recognizes termination; model distributions favor continued document-like text after short responses. EOS and chat END have different learned contexts.',
            fix='Supervise complete assistant turns including END; validate true answer-end and premature-stop behavior separately.',
            risks='Rare EOS frequency alone does not prove causality; heavily weighting END can truncate otherwise useful answers.'),
        dict(id='validation-blindspot',classification='CONFIRMED',confidence='high',symptom='Validation improves without detecting response-ending failures',
            evidence='Both independent 524,288-target validations exactly reproduce the reported CE. All 256 validation windows contain zero EOS/END targets.',
            sources=[source('LLM-Training','src/llm_training/training.py',352),historical_source],
            reproduction='validation-baseline.json, validation-final.json; actual-dataset-shift-tests.json',
            effect='The chosen validation population cannot directly measure termination loss and contains no conversational objective.',
            fix='Add a separate assistant-turn and document-boundary evaluation; retain the historical validation unchanged for comparison.',
            risks='A newly assembled development validation must not be represented as historically unseen or substitute silently for the original metric.'),
        dict(id='masked-cache-position-offset',classification='CONFIRMED',confidence='high; separate latent API defect',symptom='Masked cached forward resets default position IDs',
            evidence=f'Actual checkpoint FP32 logit maximum error {masked["original"]["max_absolute"]:.6f}; explicit past-length position IDs reduce it to {masked["explicit_offset"]["max_absolute"]:.8f}. Original generator compacts prompts and supplies no model mask.',
            sources=[expected_source,generator_source],reproduction='masked-cache-isolated-correction.json and isolated CPU regression test',
            effect='Incorrect logits for callers that pass current-token masks with cached states. Not a demonstrated cause of this run’s unmasked generation failures.',
            fix='Propose owner-component position handling correction with an explicit cache/mask API contract after validated differential tests.',
            risks='Past padding masks are not represented in the existing cache tuple; adding only an offset is not a general padded-cache solution.'),
        dict(id='weak-foundational-behavior',classification='LIKELY',confidence='medium',symptom='Failure also appears in neutral completion prompts, before long loops',
            evidence='Eight selected reference-answer argmax sequences differ at their first target; raw arithmetic, grammar and supplied-fact outputs are poor. Earlier base already had 46/48 severe outputs. Only 0.022106 CE improvement occurs in this continuation.',
            sources=[historical_source],reproduction='historical raw outputs, likelihood comparisons, token-traces.jsonl, teacher-vs-free-corpus.jsonl',
            effect='SFT may teach response format while leaving factual/reasoning weaknesses. First-token surface mismatch is not itself a semantic error: Rome has a correct answer-bearing prefix.',
            fix='Make literal task improvement and reviewed relevance/coherence required alongside termination in the bounded feasibility experiment.',
            risks='No blinded human review exists; undertraining or model size has not been causally isolated.'),
        dict(id='corpus-boilerplate-transfer',classification='LIKELY',confidence='medium-low',symptom='Generations imitate references, categories, biographies and article headings',
            evidence='Such structures occur in decoded real training windows and generations. 49/256 sample windows have a line repeated at least three times; none exceeds 0.5 word 4-gram repetition, and none exceeds 0.65 token 4-gram repetition.',
            sources=[target_source],reproduction='training-samples.jsonl and document-samples.jsonl',
            effect='Supports a document-style continuation bias, not a proven dataset of pathological loops.',
            fix='Screen boilerplate in a future data experiment only if behavioral learning remains blocked after objective testing.',
            risks='Blindly removing repeated lines or wiki categories can discard valid text; no destructive dataset edits are justified.'),
    ]
    output('root-causes.json',json.dumps(findings,indent=2))
    ranked='\n\n'.join(f"### {i+1}. {f['id']} — {f['classification']}\n\n**Confidence:** {f['confidence']}. **Symptom:** {f['symptom']}.\n\n{f['evidence']}\n\n**Generation effect:** {f['effect']}\n\n**Reproduce:** {f['reproduction']}. **Sources:** {'; '.join(f['sources'])}.\n\n**Proposed intervention:** {f['fix']} **Risk:** {f['risks']}" for i,f in enumerate(findings))
    output('DIAGNOSTIC_REPORT.md',f"""# Codexa post-training generation diagnostic

## Executive verdict

The failure primarily concerns the learned model/data objective, amplified by greedy decoding. The run is continued general-text pretraining, not conversational SFT. Its modest teacher-forced CE improvement is real: 35/48 greedy token sequences change between baseline and final, but exact-match, repetition and length-ending aggregates do not improve. Neither checkpoint demonstrates acceptable assistant behavior on the new benchmark. That last assessment is an agent observation from examples, not a completed human scoring exercise.

The historical inference path is not structurally broken in the tests performed. A separate masked-cache position defect is confirmed and isolated; the reported generator does not use that path. BF16 can alter free-generation branches around narrow logit margins. Details and the original tolerance failures are preserved in INFERENCE_AUDIT.md.

{common}

## Verified run and checkpoint

| Item | Independently verified result |
|---|---|
| Final model | `{final['path']}` |
| SHA-256 | `{preservation[0]['after']}` |
| Baseline | `{baseline['path']}`; authentic initialization SHA `{preservation[1]['after']}` |
| Model tensors | {final['model_tensor_count']}, all finite; strict architecture load; tied embeddings |
| State | Optimizer, scheduler and RNG present; no optimizer restored or constructed |
| Updates / targets | {training['updates']:,} / {training['tokens_processed']:,}; zero missing or duplicated steps |
| Weighted CE | {validation['baseline']['weighted_loss']:.8f} → {validation['final']['weighted_loss']:.8f}, exactly reproduced |
| Perplexity | {math_exp(validation['baseline']['weighted_loss']):.4f} → {math_exp(validation['final']['weighted_loss']):.4f} |
| Wikipedia / FineWeb-Edu | 3.03582136 → 3.01320028 / 3.16207209 → 3.14029351 |
| Historical exact / severe / length | 0/48 / 46/48 / 47/48 for both checkpoints |
| Expected-answer preference | 24/48 → 26/48; auxiliary two-choice likelihood, not generated correctness |
| Recorded elapsed / optimizer-step time | 10.7861 h / {training['sum_step_seconds']/3600:.4f} h |
| Final outcome | `COMPLETED / STEP_CAP`, process exit 0 |

The baseline is available and hash verified. Its original 10,000-update checkpoint records 655,360,000 targets. Reconstructing its August validation population is a separate issue: earlier investigation reports its FineWeb-only binary/index missing. This report uses the completed continuation’s recovered baseline evaluation and validation selection, not that missing historical population.

## Canonical ownership and locations

| Subsystem | Canonical implementation / inputs |
|---|---|
| Architecture | LLM-Architecture `src/llm_architecture/model.py`; actual configuration in checkpoint and frozen `config/job.json` |
| Tokenizer / vocabulary | LLM-Tokenizer `src/llm_tokenizer/tokenizer.py`; hash-bound tokenizer-base-v1/tokenizer.json in 31-LLM Source |
| Packing / conversations | LLM-Data token_data/token_views and mixed-data CLI; LLM-Tokenizer sft.py; LLM-Data chat_dataset.py for separate SFT |
| Training / loss | LLM-Training pretrain_run.py orchestration and training.py; architecture model.py CE |
| Inference | LLM-Inference generate.py / native_chat.py; installed environment module paths and hashes in manifest |
| Evaluation | Integration codexa_workspace foundation_training/foundation_diagnosis/evaluation; all original prompts recovered from JSONL |
| Completed run | outputs/training/codexa-934m-foundation-5h-v1: config, metrics, supervisor/segments, evaluations, checkpoints |
| Shared preservation | workspace-infrastructure; historical physical assets resolve through artifacts.local.json to 31-LLM Source |

## Ranked causes and associated defects

CONFIRMED labels identify directly reproduced observations or defects. The primary causal interpretation is supported by their combination and the inference exclusions; the proposed training intervention has not been tested and is not a proven cure.

{ranked}

## Possible hypotheses and ruled-out explanations

- **POSSIBLE:** insufficient foundational training, imperfect corpus filtering and semantic duplication. The known lineage has 772,800,512 targets for 934,356,480 parameters, but that ratio is not proof of causality or an adequate reason for another long run.
- **CONFIRMED numerical sensitivity:** BF16 changes two of eight selected cached-versus-full generation branches at tied candidate logits; FP32 comparisons at those branch points agree. This affects individual text but does not explain the unchanged failure aggregates through continued training.
- **RULED OUT in audited paths:** checkpoint corruption/nonfinite tensors, tokenizer hash/vocabulary mismatch, tied-head mismatch, a second target shift, ignored EOS targets, failure to recognize supplied EOS/END, and a gross unmasked KV position/causal-mask error.
- **RULED OUT for this selected run:** repeated window locations or validation window-ID overlap; semantic near-duplication across the whole corpus remains unmeasured.
- No claim that model size alone causes failure, that every sample is clean, or that the model lacks all language knowledge is justified.

## Architecture, optimization and stability

24 layers, width 1,536, 24 heads of dimension 64, FFN 6,144, 16,384 vocabulary and 2,048 learned positions; RMSNorm, SwiGLU, SDPA and tied output embeddings. RoPE is configurable but inactive in this checkpoint. The actual frozen job takes precedence over YAML templates with different warmup/max-step values.

AdamW8bit, betas 0.9/0.95, weight decay 0.1, epsilon 1e-8; BF16 autocast over FP32 weights, gradient checkpointing, microbatch 1 × accumulation 32, clipping 1.0. Warmup 20 updates, cosine 2e-5 → 2e-6. Update 1 records zero LR from the zero-based schedule; it is not an unexplained optimizer reset.

Training loss averages {training['first_64_mean_loss']:.5f} in the first 64 updates and {training['last_64_mean_loss']:.5f} in the last 64. Gradient norms span {training['gradient_norm']['minimum']:.4f}–{training['gradient_norm']['maximum']:.4f}; no nonfinite metrics or guard-scale explosions. Training and validation improve modestly; neither proves conversational adequacy or broad generalization. Three memory-headroom stops at steps 129, 152 and 671 and a deadline stop at 1,697 precede completion. All segment configuration hashes match, with contiguous update/token counters and saved optimizer step 1,792. No detected schedule/optimizer reset or effective batch change.

## Recommended next action

Prepare a small, reviewed complete-turn conversational learning feasibility experiment, not another identical general-text continuation. Require answer usefulness and factual/context consistency as well as END improvement. Earlier SFT experiments already show that improved stopping can coexist with unacceptable answers; they are historical evidence, not fresh measurements in this investigation.

No full training run is justified by these results. A specifically authorized, bounded feasibility pilot is a reasonable next hypothesis test. No pilot, optimizer update, launch, resume or notification has been executed here. See NEXT_EXPERIMENT.md.

## Reproduction and limitations

{reproduction}

All critical originals have identical before/after hashes. Full original raw source datasets and original August validation inputs have not been independently recovered here. The entire 25.6 GB corpus payload was not rehashed; the 3.3 GB index and complete selected 117.4M-target payload were independently hashed. Sampled quality/near-duplicate findings do not certify the entire corpus. Human semantic review remains pending.
""")
    table='| Model | Suite | Decode | N | Exact / scored | Severe old / independent | Length | EOS / END | Mean tokens |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
    table+='\n'.join(f"| {x['checkpoint']} | {x['suite']} | {x['decode']} | {x['cases']} | {x['exact_surface_matches']}/{x['exact_scored_cases']} | {x['historical_severe']}/{x['independent_severe']} | {x['length']} | {x['EOS']}/{x['END']} | {x['mean_generated_tokens']:.1f} |" for x in metrics)
    comparison_text=f"""# Generation comparison

{common}

## Matched measurements

400 newly generated responses: baseline 48 historical + 32 conversational, final 288 historical decoding variants + 32 conversational. Each checkpoint reproduces its own 48/48 historical greedy token arrays exactly. Comparing baseline with final, {identical_pairs}/48 arrays are identical and {48-identical_pairs}/48 change; five first tokens change. The unchanged failure aggregates do not mean every output is unchanged. Paired records are in baseline-final-paired.json.

{table}

Historical severe is word 4-gram repetition >0.5. Independent severe additionally considers token 4-gram repetition >0.65 or a nonempty line repeated three times. Both are mechanical screens, not semantic judgments. Exact scores apply to all 48 historical references but only seven of the sixteen conversation prompts. A surface mismatch does not automatically mean a semantically wrong answer.

## Controlled settings

All generations use seed 42, maximum 128 new tokens, EOS 2, PAD 0 and stop sequence [7], with no automatic BOS insertion. No no-repeat-gram ban or shortened generation cap is used.

| Variant | Sampling | Temperature | Top-p | Top-k | Penalty | Compare against |
|---|---|---:|---:|---:|---:|---|
| greedy | off | unset | unset | unset | 1.0 | historical reference |
| sample-t1 | on | 1.0 | unset | unset | 1.0 | greedy: selection policy |
| sample-t08 | on | 0.8 | unset | unset | 1.0 | sample-t1: temperature |
| nucleus | on | 0.8 | 0.9 | unset | 1.0 | sample-t08: top-p |
| topk | on | 0.8 | unset | 40 | 1.0 | sample-t08: top-k |
| nucleus-r11 | on | 0.8 | 0.9 | unset | 1.1 | nucleus: penalty |

The penalty includes prompt IDs, as the existing engine implements it. Sampling reduces loops but does not make the responses reliably relevant. Repetition metrics can miss semantically repetitive paraphrases: “READY” and similar ideas repeat in sampled output without crossing the severe threshold. Full raw token/text records remain unchanged in generations-*.jsonl.

## Conversational human-review candidates

The following excerpts are original raw decoded strings, including initial spaces and any special stop tokens. Scores for coherence, relevance, following instructions, repetition, termination and overall usefulness are **pending human review**. Candidate-specific automated metrics are in generation-metrics.csv; no automatic truthfulness rating is substituted.
"""
    index={(x['checkpoint'],x['suite'],x['decode'],x['id']):x for x in generations}
    for case in read('conversational-cases.json'):
        comparison_text+=f"\n### {case['id']} — {case['category']}\n\nPrompt: {case['question']}\n"
        for label in ('baseline','final'):
            for decode in ('greedy','nucleus'):
                row=index[(label,'conversation',decode,case['id'])]
                comparison_text+=f"\n**{label}, {decode}:** {row['generated_tokens']} tokens; finish `{row['termination']}`; independent severe `{row['metrics']['independent_severe']}`.\n\n"+excerpt(row['raw_text'])
    comparison_text+='\n## Representative historical comparison\n\n'
    for case_id in ('foundation-00-neutral','foundation-04-neutral','foundation-06-direct','foundation-18-direct'):
        comparison_text+=f"\n### {case_id}\n\n"
        for label in ('baseline','final'):
            row=index[(label,'historical','greedy',case_id)]
            comparison_text+=f"**{label}; prompt:** {row['prompt']}\n\n"+excerpt(row['raw_text'])
    comparison_text+='\nAgent observations: Rome appears in the correct first sentence for the Italy completion, followed by invented historical claims and loops. Numerical references often become “0.001” rather than the requested calculation. Chat prompts yield article fragments, lists or paraphrases of the prompt. These observations identify review priorities and do not constitute independent human ratings.\n'
    output('GENERATION_COMPARISON.md',comparison_text)
    output('TOKENIZER_EOS_AUDIT.md',f"""# Tokenizer and EOS audit

{common}

## Token identity and compatibility

| ID | Token | Intended use |
|---:|---|---|
| 0 | `<pad>` | Padding; absent from all current training targets |
| 1 | `<bos>` | Explicit only; automatic BOS/EOS insertion disabled |
| 2 | `<eos>` | Document termination |
| 3 | `<unk>` | Unknown; absent from all current training targets |
| 4 | `<\\|system\\|>` | Chat system role |
| 5 | `<\\|user\\|>` | Chat user role |
| 6 | `<\\|assistant\\|>` | Open assistant role |
| 7 | `<\\|end\\|>` | Completed chat turn |

Tokenizer SHA `{preservation[2]['after']}`; vocabulary, embeddings and head all 16,384. Both checkpoints reference the same tokenizer hash. 256 sampled training inputs round-trip without any failure. Prompt reconstruction agrees with all 48 frozen historical input arrays. Token IDs and tokenizer files were never changed.

## Training targets and boundaries

The complete selected target scan finds exactly {target_audit['special_token_counts']['2']:,} EOS targets ({100*target_audit['special_token_counts']['2']/target_audit['targets']:.4f}% of targets) and no IDs 0, 1, 3, 4, 5, 6 or 7. No PAD is trained as text in this run. EOS is not masked: real dataset fetches match the stored shift, and model CE ignores only -100. 878 sampled document-boundary observations end with EOS; zero failures. These observations include documents repeated across adjacent inspected windows and are not 878 unique documents.

Packed documents can share attention, with one EOS and no position reset. This is the declared pretraining policy, not an accidentally missing boundary. Inputs/labels shift once in {target_source}; model CE does not shift again in {loss_source}.

Both current validation selections contain zero EOS/END targets. Their loss improvement says nothing directly about response-end prediction. All zero end-loss counts are reported as unavailable, not zero NLL.

## Termination probabilities and actual stop behavior

Eight reference-answer endings assign EOS probability {eos_min:.6g}–{eos_max:.6g}; END maximum {end_max:.6g}. token-traces.jsonl preserves every selected free-generation step, EOS/END rank and probability, entropy, top-five candidates, selected token probabilities and first sustained repetition position. Short answer surfaces are diagnostic contexts, not necessarily grammatical full-document endings.

{eos_table}

Thirty-two real document tails per checkpoint show different behavior: EOS mean probability {statistics.mean(x['distribution']['eos_probability'] for x in doc_probes['baseline']):.6f} baseline → {statistics.mean(x['distribution']['eos_probability'] for x in doc_probes['final']):.6f} final; EOS rank 1 on 5/32 for each. This demonstrates contextual EOS learning and contradicts a claim that EOS is universally unlearned. It does not show reliable assistant-turn termination.

Forced EOS and END stop immediately on CPU and CUDA, preserve the terminator in raw IDs, omit it from visible output/callbacks, and record the correct stop cause. Multi-token stops and padded rows also pass. Actual FP32 model generation matches CPU/GPU for the eight-token bounded cross-device check; this is not a promise of device-identical sampling.

No END is generated in any of 400 new responses. Historical 47/48 length endings arise because the model does not choose a recognized stop, rather than an engine ignoring a chosen EOS.

## Conversational serializer

{source('LLM-Tokenizer','src/llm_tokenizer/sft.py',53)} leaves an assistant marker open for inference. Complete training turns include END, supervise assistant content plus END, mask roles/user/system, and reject overlong whole conversations. The chat dataset shifts these labels once: {source('LLM-Data','src/llm_data/chat_dataset.py',90)}. Sixty-four existing records from each of two local datasets serialize successfully; example masks are preserved. These datasets were not inputs to this completed run.

Recommended correction is teaching a complete assistant turn in a bounded SFT feasibility experiment, not redefining EOS or substituting a short generation cap. EOS weighting and END weighting are separate hypotheses and remain unapproved.
""")
    output('INFERENCE_AUDIT.md',f"""# Inference audit

{common}

## Loading and historical reproduction

Strict native model-state loading, matching architecture and tokenizer hashes, all finite tensors, tied embeddings, evaluation mode, inference mode, FP32 weights and BF16 CUDA autocast. Original context is 2,048 learned positions; actual prompts are unpadded single rows. Historical generation uses cache, greedy selection, max 128, seed 42, penalty 1, EOS 2 and stop [7]. 96/96 historical token arrays reproduced.

Installed inference SHA `{manifest['code']['llm_inference.generate']['sha256']}` differs from sibling source SHA `{manifest['code']['llm_inference.generate']['source_sha256']}`. The entire diff is the source's additional finite-logit guard. The frozen evaluator also installs a finite-logit hook. No historical inference hash was frozen; exact output reproduction supports behavioral reconstruction rather than pretending an unavailable historical hash exists.

## KV equivalence and numerical precision

15 initial same-prefix comparisons: eleven BF16 and four FP32, prefix lengths 1/8/32/128/2040/2047, chunks 1/4. All preserve argmax. FP32 short-prefix maximum error is {max(x['max_absolute'] for x in comparisons if x['precision']=='fp32'):.8f}; all short checks pass atol/rtol 1e-4. BF16 normally has max error 0.0625; the 2040+4 chunk test reaches 0.33203125 and fails the original atol .125/rtol .02 screen. Raw relative errors near zero logits can be large; normalized RMS and KL accompany them.

The long-prefix FP32 follow-up reaches {long_checks[0]['max_absolute']:.8f}, failing the original strict 1e-4 screen, with normalized RMS {long_checks[0]['normalized_rms']:.8g}, KL {long_checks[0]['kl']:.8g} and identical argmax. Layerwise localization preserves the failure rather than silently raising a tolerance: learned positions are exactly 2040–2043 in both paths, small numerical differences accumulate across layers, and a separately reported long-context atol 5e-4/rtol 1e-4 screen passes. See precision-localization.json. These measurements support roundoff sensitivity; they do not justify bitwise-equivalence claims.

Uncached full-sequence generation matches cached generation on {sum(x['identical'] for x in cache_generation)}/8 selected prompts. The two BF16 branches occur at steps 25 and 21. Actual accumulated-cache branch distributions and FP32 comparisons are preserved in precision-localization.json, rather than rebuilding only a fresh prefill cache. Neither branch recovers a useful answer; both cached and uncached paths degenerate.

At both branch points the actual BF16 cached top-two margin is zero and the direct margin is 0.0625, with different argmax tie selection. Repeating identical conditioning prefixes in FP32 yields matching argmax and maximum differences 0.00007427 and 0.00004983. This directly localizes those branches to reduced-precision sensitivity.

Two real padded batch rows match single-prompt output. Generation removes padding row-by-row, then calls the model without a mask. CPU synthetic learned/rotary cache, context exhaustion and masking tests pass. Real CPU/GPU FP32 logits differ by {read('actual-cpu-gpu.json')['both_fp32']['max_absolute']:.8f}, with the same eight generated IDs. RoPE is not active in the real checkpoint.

## Confirmed separate masked-cache defect

Default positions with a supplied mask use a fresh cumsum and omit past length at {expected_source}. For the same actual checkpoint prefix, masked cached logits differ by {masked['original']['max_absolute']:.6f}; specifying position_ids equal to the cached length reduces error to {masked['explicit_offset']['max_absolute']:.8f}. A tiny CPU differential regression demonstrates the same defect and correction.

The historical generator at {generator_source} supplies no model mask, and its normal offsets are correct. This defect is therefore **confirmed but ruled out as the demonstrated explanation for these original failures**. The diagnostic correction changes only call arguments in an isolated model. No production patch was applied. A future component correction must also define how past padding masks are retained; a simple offset alone does not solve arbitrary padded caches.

## Sampling and stopping

Top-p/top-k filtering, sign-aware repetition penalty, finite softmax, fixed-seed selection, multi-token stop tracking and context caps pass the selected generation regression suite. Greedy ignores temperature/top-p/top-k when sampling is disabled, as configured. No repeat-ngram ban is enabled. Forced EOS/END recognition works on both devices. Context is capped rather than silently sliding learned positions; the tested 2,048 boundary is respected.

## Teacher-forced versus autoregressive behavior

Eight diagnostic expected-answer targets differ from argmax at the first token. This is a surface-distribution test: a correct semantic alternative such as “the city of Rome” need not start with the reference token “Rome.” Raw examples nonetheless show substantive arithmetic/context failures. Sustained word 4-gram repetition starts at steps {', '.join(str(x['first_sustained_repetition']) for x in traces)} for these selected outputs. Four real validation-text continuations compare target probabilities and teacher/free argmax at each step in teacher-vs-free-corpus.jsonl. Once prefixes differ, these are different conditioning histories, not an inference-equivalence test.

No production weight, generation configuration, architecture, or tokenizer change is warranted as a quick fix for the reported behavior. The masked-cache owner fix is reviewable separately; the learned objective and behavioral validation require the primary corrective experiment.
""")
    output('DATASET_AUDIT.md',f"""# Dataset audit

{common}

## Actual completed-run exposure

The frozen run uses the existing mixed uint16 corpus at `{json.loads((RUN/'config/job.json').read_text())['spec']['data']['corpus_path']}`. The corpus contains 12,819,847,234 declared tokens and 13,383,283 indexed document occurrences. The full 3.3 GB index was streamed and its SHA independently matches the manifest: `{dataset['streamed_index_sha256']}`.

Exact source attribution of every selected target: FineWeb-Edu 71,696,026 ({100*71696026/117440512:.4f}%), Wikipedia 45,744,486 ({100*45744486/117440512:.4f}%). Zero conversational examples. Original loss weighting uses corpus shares 61.0997%/38.9003%, slightly different from this particular sampled exposure; validation has 128 windows from each source before weighting.

Validation selection document intervals are checked against actual target windows in the additional validation-split-audit.json. Its window-ID disjointness is a separate guarantee from normalized/semantic content disjointness; that latter property is not certified for the complete corpus.

57,344 distinct window locations, each used once; zero validation window-ID overlap. This excludes repeated location sampling, not duplicate semantic content. Full selected target payload SHA `{target_audit['selected_target_payload_sha256']}` binds actual frozen-order target slices. No target lies outside vocabulary; maximum ID {target_audit['maximum_id']}. Full corpus checksum is recorded historically but not refreshed for all 25.6 GB.

## Formatting and quality

Read-only sample: 256 uniformly sampled actual windows, seed 42, including documents from both sources. It was not pre-stratified by source; observed source document counts are 433 Wikipedia and 445 FineWeb-Edu boundary observations. Full decoded inputs, targets, masks and document provenance are saved. Thirty-two representative full-document tail/partial body records, sixteen per source, are separately preserved; any 8,192-token truncation is marked.

All 256 input token sequences round-trip. 878 observed EOS boundaries pass. Sample targets contain 623 EOS and no PAD/UNK/role/END. All 2048 targets in each base window are supervised, with a single next-token shift and no padding. Eight real dataset class fetches independently match stored input/target sequences.

Zero exact normalized duplicate-window pairs and zero >=0.8 five-word shingle Jaccard pairs in the sample. These are small-window screens, not corpus-wide deduplication. Forty-nine windows repeat a nonempty line three times, often because packed Wikipedia articles repeat “References”, “External links” or category structures. **None** exceeds 0.5 repeated word 4-grams or 0.65 repeated token 4-grams. Calling all 49 pathological training loops would be incorrect. Real samples nevertheless support a document/boilerplate-style continuation explanation for observed outputs.

No empty source records or malformed raw JSON were detected in the inspected index/data examples. Complete original raw Wikipedia/FineWeb source stores were not recovered; decoded tokenized documents cannot certify raw encoding/preprocessing for the entire historical source population. Original raw-record defects, language quality, factual correctness and full semantic near-duplication remain incompletely measured. Source text examples include legitimate foreign quotations; non-ASCII alone is not corruption.

## Objective suitability and local conversational data

This is general-text pretraining, with attention crossing EOS document boundaries by declared policy. It provides no positive assistant-END or role-format training in these 117M targets. Conversational behavior cannot be inferred from its validation loss. It also does not explain away weak ordinary completions; those must be evaluated directly.

General Chat v2 is available: 182,892 training, 10,135 validation and 9,874 test records, sourced from UltraChat and OASST1. Its manifest describes root/substantial-message duplicate grouping and historical exposure; it does not certify factual correctness or semantic paraphrase isolation. Current first-64 checks for both General Chat v2 and the repair dataset serialize successfully with complete assistant END and role masking. These are structural checks, not a quality admission decision.

The repair dataset has 9,627 training records and 2,469 validation records, but prior investigation documents overlapping/repeated anchors and weak coverage of exact short-response instructions. It must not be treated as automatically suitable for a fresh pilot. The separate earlier SFT ablations improved END while leaving objective entry gates failed. That historical report is context, not newly rerun evidence.

For a future pilot, use reviewed, complete short turns from existing General Chat v2 with benchmark contamination checks, keeping new split provenance explicit. Do not import or alter data now. No dataset, tokenizer, vocabulary or preprocessing pipeline was changed.
""")
    output('GPU_MEMORY_AUDIT.md',f"""# GPU memory and run stability audit

{common}

## Actual hardware and measured diagnostics

RTX 4080, 16,376 MiB physical VRAM. Initial snapshot: approximately 14,190 MiB free and 23 GiB available RAM; the exact contemporaneous preparation snapshot is in manifest.json. It records compute-client names; this is not a historical process inventory at each trainer failure.

Sequential CPU checkpoint mappings load only model weights onto CUDA. Diagnostic peak allocated {peak_allocated/1024**3:.3f} GiB, reserved {peak_reserved/1024**3:.3f} GiB across saved measurements. The 8 GiB allocation ceiling and 2 GiB physical headroom guard were never breached. Near-context FP32 math, full-sequence versus cache tests and a CPU transfer are included. No training activation or optimizer allocation was reproduced.

## Historical training measurements

Peak allocated {training['peak_allocated_vram_bytes']['maximum']/1024**3:.3f} GiB; peak reserved {training['peak_reserved_vram_bytes']['maximum']/1024**3:.3f} GiB. Reserved minus allocated is allocator caching plus fragmentation/other scope effects; this gap alone does not prove fragmentation. Microbatch 1, sequence 2048, accumulation 32, BF16, 8-bit optimizer, activation checkpointing, SDPA and default allocator remained unchanged across continuations.

At the highest recorded reserved usage, the 16 GiB card has only about 3.7 GiB outside trainer reservations. The frozen 1.5 GiB free-memory guard leaves roughly 2 GiB for display/other GPU clients. Three stops at 129, 152 and 671 explicitly report MEMORY_HEADROOM_GATE; this is distinct from a CUDA OOM crash. The contemporaneous project record attributes the first stop to approximately 3 GB of other-client use with a stable 12,420 MiB trainer reservation. That attribution is supported by the local record but an independent per-process failure-time trace is unavailable.

Memory checks use actual driver free memory at optimizer boundaries: {source('LLM-Training','src/llm_training/pretrain_run.py',365)}. Gradient accumulation does not keep 32 activation graphs resident. Historical phase-specific activation allocations and checkpoint-loading spikes were not logged finely enough to isolate them. No claims of exact activation-memory consumption are made.

## Resume and elapsed accounting

All interrupted segments share configuration SHA 83157775021700dd5853ef60c1fd7a0e30679dcf6ad616e25725bc55c623327b. Metric steps 1–1792 are present once, and selected windows are unique. Three memory stops and one deadline stop precede an explicit extension from step 1697. The extension changed deadline accounting, not the frozen model/batch/sequence/optimizer specification. Final optimizer states record update 1792.

Recorded elapsed budget is 38,830.03 seconds. Sum of optimizer-step metrics is {training['sum_step_seconds']:.2f} seconds; it excludes validation, checkpoint writes, generation and stopped time. Do not bill 10.79 h as GPU training compute.

## Practical recommendations

Retain the existing headroom gates and default allocator. Close optional GPU applications before an explicitly authorized trainer launch, keep the required existing dashboard, and avoid opening extra GPU windows. Do not kill other processes automatically.

A proposed 512-context diagnostic SFT pilot should fit the same 16 GiB hardware more conservatively than the measured 2048-context trainer, but its actual optimizer/activation peak has not been measured here. Require the normal 13,928 MiB startup free-VRAM, 16 GiB available-RAM, disk-reserve and viewer/supervisor gates. Use microbatch 1 and complete short turns. Any allocator, attention-backend or checkpoint-loading optimization needs a separate controlled measurement; none was permanently changed.
""")
    output('NEXT_EXPERIMENT.md',f"""# Next experiment: complete-turn conversational learning feasibility

## Primary hypothesis and authorization boundary

The verified final checkpoint can learn to answer and end complete assistant turns when supplied the currently absent conversational objective. This is a **feasibility hypothesis**, not a prediction that a short SFT stage will create a useful general assistant. Plain-completion weaknesses and failed historical SFT entry gates make a long run unjustified.

**Primary intervention:** assistant-only chat-v1 SFT on a small, reviewed set of complete short conversational turns, including supervised END. Keep token definitions, architecture, and verified checkpoint unchanged. Do not combine this pilot with EOS reweighting, changed decoding, corpus cleanup, larger context or allocator experiments.

**Not executed or authorized:** dataset admission/preparation, optimizer construction/updates, trainer launch/resume, production model promotion. This document does not constitute approval. No five- or ten-hour run is recommended.

## Frozen pilot proposal

- Initialization: verified final model `{final['path']}`, SHA `{preservation[0]['after']}`; fresh optimizer/scheduler in a new namespace, never mistaken for exact training resume.
- Data: 512 reviewed complete conversations, each at most 512 serialized tokens: 224 OASST1 and 224 UltraChat turns from existing General Chat v2, plus 64 independently authored short-format examples to be prepared only after separate authorization. The 64 examples cover 16 each of word-only commands, list formatting, simple arithmetic with varied distinct targets, and supplied-context answers. This addresses the prior natural-chat pilot's demonstrated lack of constrained-response practice. No existing repair anchors, evaluation prompts or evaluation-specific answers are copied. Filter exact normalized questions and substantial eight-word content overlap with the benchmark; common factual vocabulary alone is not leakage. Freeze provenance and split policies before approval. No partial assistant truncation.
- Sampling: 128 maximum optimizer updates, microbatch 1, accumulation 8, seed 73. Traverse the 512 distinct conversations once in seeded order, then once again with a separately seeded order. Exactly 1,024 presentations if the full cap is reached. Repetition is explicit in this tiny feasibility design, not evidence of broad generalization.
- Tokens: at most 524,288 context positions. Actual assistant-supervised tokens are admission-dependent and must be counted and reported; assume 40–70% (approximately 0.21–0.37M) solely for budgeting. Freeze the selected IDs/hashes and exact supervised count before any requested approval.
- Objective: existing correctly shifted assistant content + END targets, ignored user/system/role/padding labels; unweighted per-supervised-token CE. No tokenizer changes, EOS weighting, END weighting, or altered loss denominator.
- Optimization: existing AdamW8bit, LR 5e-6, warmup 8, cosine to 5e-7, betas 0.9/0.95, epsilon 1e-8, weight decay 0.1, clip 1.0, BF16/FP32 weights, activation checkpointing. These are conservative test defaults, not demonstrated optimal hyperparameters.
- Evaluate at 0, 32, 64 and 128 updates; checkpoint at 64 and final into separate artifacts. Retain the verified reference. Use the 48 historical prompts plus sixteen conversational cases, identical greedy/nucleus settings and maximum 128 tokens, without any penalty increase.
- Add 32 reviewed, source-balanced assistant-turn validation conversations disjoint by root/normalized substantial text from the pilot. Include gold answer-end END loss and premature-END probabilities. Mark historical exposure caveats; do not call these newly assembled development cases a protected acceptance benchmark.

## Acceptance, failure and rollback

For the fixed sixteen conversational development prompts, require all of:

1. At least 8/16 outputs naturally end, and at least 6/16 end with chat END rather than falling into article EOS.
2. No more than 4/16 independent severe-degeneration flags.
3. At least 3/7 exact constrained-reference successes, with no silent case folding or output cleaning for case-sensitive instructions.
4. A real blinded human reviewer accepts relevance/coherence and factual/context consistency on at least 8/16, with at least four paired improvements and no more than one clear paired regression. This is a small pilot gate, not a statistically established capability claim.
5. Mean gold assistant-END NLL improves by at least 0.5 from this pilot’s step-zero evaluation, while premature END probability on selected non-ending assistant positions does not rise by more than 0.02 absolute.
6. Neither historical validation source CE rises by more than 0.1; all tensors/logits remain finite and normal resource gates hold.

Stop immediately on nonfinite/OOM, or follow the existing supervisor’s headroom stop policy. Stop after two evaluation points with worsening degeneration or reference-source CE over the retention limit. END/CE improvement without the usefulness gates is **failure** of the conversational-learning hypothesis at this exposure; it is not permission to extend the run. If the 128-update cap fails, do not automatically increase hours/steps or prescribe more SFT. Reassess foundational knowledge, admitted examples and actual gradients with the saved evidence.

Rollback means discarding promotion of the pilot and retaining the original verified checkpoint/production selection. Keep pilot artifacts for evidence; delete nothing and overwrite nothing. Any production masked-cache correction is a separate owner-component change validated by the isolated regression; it is not bundled into this learning experiment.

## Compute and memory budget

Latest real full-context throughput median {training['tokens_per_second']['median']:.0f} targets/s gives a nominal 65-second context-token floor for 524,288 positions; dynamic shorter-sequence and update overhead make that an estimate, not a measurement. Allow roughly 2–5 minutes for optimizer work, plus 6–10 minutes for baseline/intermediate/final generations, teacher-forced validation, loading, hashing and two recovery publications. Use a **15-minute total supervised-job cap with 180 seconds reserved for finalization**, stopping at 128 updates or the earlier time bound. This is a bounded mechanism test, not a wall-time target to fill.

Hardware: existing RTX 4080 16 GiB. Conservatively reserve up to the observed full-context trainer footprint, approximately 12.3 GiB; exact 512-context pilot peak is unavailable. Require normal startup free VRAM 13,928 MiB, 16 GiB available host RAM, 180 GiB free disk and the 100 GiB recovery reserve. No gates may be lowered. The read-only dashboard and OS supervisor are required for any separately authorized launch. If startup/profile evidence cannot meet these gates, the pilot remains blocked; do not change production memory behavior automatically.

Before authorization, make the selected data, exact token accounting, frozen job, expected storage, independent review queue and reviewable gate results concrete. Ask for approval only after preparation, and never launch from this diagnostic assignment.
""")
    review=[]
    for row in generations:
        if row['suite']=='conversation':
            review.append(dict(checkpoint=row['checkpoint'],id=row['id'],decode=row['decode'],prompt=row['prompt'],raw_text=row['raw_text'],
                automated_metrics=row['metrics'],scores={k:None for k in ('coherence','relevance','instruction_following','repetition','termination','usefulness')},reviewer=None))
    output('human-review-candidates.jsonl','\n'.join(json.dumps(x,ensure_ascii=False) for x in review))
    shuffled=review.copy();random.Random(42).shuffle(shuffled);blinded=[];key={}
    for i,row in enumerate(shuffled):
        candidate=f'candidate-{i:03}'
        key[candidate]=dict(checkpoint=row['checkpoint'],case_id=row['id'],decode=row['decode'])
        blinded.append(dict(candidate=candidate,prompt=row['prompt'],raw_text=row['raw_text'],
            rubric='For each dimension: 0 fails, 1 limited, 2 satisfactory, 3 strong. Verify factual/context claims independently; list evidence.',
            human_scores=row['scores'],reviewer=None,verification_evidence=[]))
    output('human-review-blinded.jsonl','\n'.join(json.dumps(x,ensure_ascii=False) for x in blinded))
    output('human-review-key.json',json.dumps(key,indent=2))
    tests=dict(diagnostic_unit='11 passed',existing_generation='6 passed, 1 deselected',
        historical_reproduction='96/96 exact token arrays',validation='both 524,288-target weighted CEs reproduced exactly',
        confirmed_expected_failure='masked-cache positions; isolated explicit-offset correction passes',
        numerical_screening_failures='BF16 2040+4 and strict FP32 2040+4 retained; precision localization completed',
        skipped=['existing checkpoint-CLI test to avoid optimizer-bearing fixture creation','vocabulary-training tests; production tokenizer preserved'],
        blocked=['human semantic ratings: no independent reviewer','full original raw corpus and original August validation recovery'],
        diagnostic_error_corrected='Initial baseline validation parsed source-map schema incorrectly; original failure log retained. Generation rows reused on retry. Flat loss layout initially differed by ~3.2e-5; matching trainer transpose layout reproduces CE exactly.')
    output('testing.json',json.dumps(tests,indent=2))
    output('README.md',f"# Diagnostic artifacts\n\n{common}\n\n"+'\n'.join(f'- [{name}]({name})' for name in ['DIAGNOSTIC_REPORT.md','GENERATION_COMPARISON.md','TOKENIZER_EOS_AUDIT.md','INFERENCE_AUDIT.md','DATASET_AUDIT.md','GPU_MEMORY_AUDIT.md','NEXT_EXPERIMENT.md','generation-summary.csv','generation-metrics.csv','generations-baseline.jsonl','generations-final.jsonl','root-causes.json','testing.json','preservation-final.json','training-curves.png'])+'\n\nRebuild reports with `scripts/render_post_training_generation_diagnostic.py --directory <this-directory>` using the existing integration virtualenv. The runner, renderer and focused tests live in the integration repository. `build_reports.py` retains the earlier local renderer. Phase attempts and logs preserve errors and numerical failures. All diagnostic evidence is ignored by Git.\n')
    for name in ['DIAGNOSTIC_REPORT.md','GENERATION_COMPARISON.md','TOKENIZER_EOS_AUDIT.md','INFERENCE_AUDIT.md','DATASET_AUDIT.md','GPU_MEMORY_AUDIT.md','NEXT_EXPERIMENT.md']:
        assert (D/name).stat().st_size>1000
    print('Reports built:',D)


def math_exp(value):
    import math
    return math.exp(value)


def plot_training_curves():
    """Use existing matplotlib where available; never install dependencies."""
    records=[json.loads(s) for s in (RUN/'metrics/train_metrics.jsonl').read_text().splitlines()]
    with (D/'training-metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    if (D/'training-curves.png').exists():return
    plotter=shutil.which('python') or sys.executable
    code='''import json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
run,directory=map(Path,sys.argv[1:])
records=[json.loads(s) for s in (run/'metrics/train_metrics.jsonl').read_text().splitlines()]
validation=json.loads((run/'metrics/report.json').read_text())['counters']['validations']
fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
steps=[x['optimizer_step'] for x in records]
axes[0,0].plot(steps,[x['training_loss'] for x in records],alpha=.2,label='per update')
smooth=[sum(x['training_loss'] for x in records[max(0,i-31):i+1])/min(i+1,32) for i in range(len(records))]
axes[0,0].plot(steps,smooth,label='32-update mean');axes[0,0].set_ylabel('Training CE');axes[0,0].legend()
for key in ['weighted_loss','wikipedia','fineweb_edu']:
 axes[0,1].plot([x['optimizer_step'] for x in validation],[x['weighted_loss'] if key=='weighted_loss' else x['sources'][key] for x in validation],label=key)
axes[0,1].set_ylabel('Validation CE');axes[0,1].legend()
axes[1,0].plot(steps,[x['learning_rate'] for x in records]);axes[1,0].set_ylabel('Learning rate')
for key in ['peak_allocated_vram_bytes','peak_reserved_vram_bytes']:
 axes[1,1].plot(steps,[x[key]/1024**3 for x in records],label=key.replace('_vram_bytes',''))
axes[1,1].set_ylabel('Trainer CUDA memory (GiB)');axes[1,1].legend()
for ax in axes.flat:ax.set_xlabel('Optimizer update');ax.grid(alpha=.2)
fig.suptitle('Codexa: complete 1,792-update record; no new training')
fig.savefig(directory/'training-curves.png',dpi=160);fig.savefig(directory/'training-curves.svg');plt.close(fig)
'''
    script=D/'plot_training_curves.py';script.write_text(code)
    result=subprocess.run([plotter,str(script),str(RUN),str(D)],capture_output=True,text=True,timeout=120)
    output('plot-result.json',json.dumps(dict(command=[plotter,str(script),str(RUN),str(D)],exit_code=result.returncode,
        stdout=result.stdout,stderr=result.stderr,dependencies_installed=False),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    D=args.directory.resolve()
    allowed=(ROOT/'outputs/diagnostics/post-training-generation-investigation').resolve()
    if not D.is_relative_to(allowed) or D==allowed:parser.error('Use a dedicated diagnostic child directory')
    from organize_post_training_generation_diagnostic import REPORTS, render
    organized = D / 'evidence/verification/organization-manifest.json'
    if not organized.exists() and not all((D/name).is_file() for name in REPORTS):
        # Fresh evaluations retain the original evidence calculation, then consolidate.
        main()
    result = render(D)
    print('Consolidated report:', D / 'REPORT.md')
    print(json.dumps(result, indent=2))
