"""CPU-safe preparation and separately approved inference-only comparisons."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import random
import sys
import time

import yaml

from llm_architecture.checkpointing import file_sha256
from llm_training.config import load_config
from llm_training.session import atomic_document
from llm_tokenizer.tokenizer import load_tokenizer
from .assets import workspace_root, resolve_input, output_root
from .behavioral_readiness import development_suite, acceptance_suite, suite_sha256
from .chat_preparation import hardware, experiment_environment

TOKENIZER_SHA256='a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e'
NATIVE_BYTES=5636070993


def stratified_validation_indices(path: Path, *, per_source: int = 32, seed: int = 42) -> list[int]:
    """Use fixed distinct-root samples per source, without declaring them historically clean."""
    pools={}
    with path.open() as stream:
        for index,line in enumerate(stream):
            row=json.loads(line)
            pools.setdefault(row['source'],[]).append((index,row['root_id']))
    rng=random.Random(seed); selected=[]
    for source,rows in sorted(pools.items()):
        rng.shuffle(rows); roots=set()
        for index,root in rows:
            if root not in roots:
                selected.append(index); roots.add(root)
                if len(roots)==per_source:
                    break
        if len(roots)!=per_source:
            raise ValueError('Validation source lacks enough independent roots: '+source)
    return sorted(selected)


def candidate_paths(root: Path) -> dict[str,Path]:
    """Only protected surviving candidates; retired pilot payloads are excluded."""
    from llm_training.recovery import RecoveryStore
    return dict(base=resolve_input(root,'checkpoints/codexa-900m-base-v1/latest.pt'),
        repair=resolve_input(root,'checkpoints/codexa-900m-sft-repair-v1/latest.pt'),
        balanced=resolve_input(root,'checkpoints/codexa-900m-sft-balanced-v1/latest.pt'),
        original_2h=RecoveryStore(root/'outputs/chat-training/codexa-900m-chat-2h-v1/checkpoints').resolve('final'),
        END3=RecoveryStore(root/'outputs/investigation/chat-regression-20261010/pilots/repair-proportional-END3/checkpoints').resolve('final'))


def resource_preflight(root: Path) -> dict:
    """Observe unchanged production gates; never free memory or modify processes."""
    observed=hardware(); free=shutil.disk_usage(root).free
    ram=observed['host_memory_bytes']['MemAvailable']
    return dict(free_bytes=free,reserve_bytes=180*1024**3,surplus_bytes=free-180*1024**3,
        disk_gate_passed=free>=180*1024**3,host_gate_passed=ram>=16*1024**3,
        gpu_gate_passed=observed['free_gpu_bytes']>=11771314176,hardware=observed,
        qualification_peak_bytes=6*NATIVE_BYTES+1024**3,
        legacy_training_peak_bytes=7*NATIVE_BYTES+1024**3,
        execution_approval_present=False)


def protected_metadata(root: Path) -> dict:
    """Compare surviving protected identities with the recorded cleanup baseline."""
    record=root/'outputs/storage-cleanup/20261010-controlled-01/before.json'
    baseline=json.loads(record.read_text())['protected']
    changed=[]
    for name,expected in baseline.items():
        path=Path(name)
        try:
            stat=path.stat()
            actual=dict(device=stat.st_dev,inode=stat.st_ino,bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)
            if actual!=expected:
                changed.append(dict(path=name,expected=expected,actual=actual))
        except OSError as error:
            changed.append(dict(path=name,error=str(error)))
    return dict(checked=len(baseline),changes=changed,
                method='device/inode/size/mtime/ctime; not a content checksum')


def build_preparation(root: Path, directory: Path) -> dict:
    """Create isolated evaluation/data proposals; never construct a full model."""
    if Path(sys.prefix).resolve()!=(root/'.venv').resolve():
        raise RuntimeError('Use the integration generative environment.')
    if directory.exists():
        raise FileExistsError('Preparation requires a fresh output namespace.')
    if not directory.resolve().is_relative_to(output_root(root).resolve()):
        raise ValueError('Preparation artifacts must use the integration output root.')
    resources=resource_preflight(root)
    if not resources['disk_gate_passed']:
        raise RuntimeError('Production disk reserve fails; no artifact generation.')
    directory.mkdir(parents=True)
    tokenizer_path=resolve_input(root,'checkpoints/tokenizer-base-v1/tokenizer.json')
    if file_sha256(tokenizer_path)!=TOKENIZER_SHA256:
        raise ValueError('Tokenizer identity mismatch.')
    tokenizer=load_tokenizer(tokenizer_path)
    dev,accept=development_suite(),acceptance_suite()
    atomic_document(directory/'development-suite.json',dev)
    # Separate file: never loaded by the development evaluator or data builder.
    atomic_document(directory/'acceptance-suite.sealed.json',accept)
    atomic_document(directory/'suite-manifest.json',dict(format='codexa-evaluation-suites-v2',
        development_sha256=suite_sha256(dev),acceptance_sha256=suite_sha256(accept),
        development_cases=len(dev),acceptance_cases=len(accept),
        acceptance_access='post-training only; no current acceptance execution',
        semantic_review_required=True,prompt_reference_review_complete=False,
        historical_exposure='pending lineage audit, never presumed clean'))
    from llm_data.curated_chat import write_curated
    curated=write_curated(directory/'curated/train.proposed.jsonl',tokenizer)
    atomic_document(directory/'curated/dataset_manifest.json',curated)
    general=resolve_input(root,'data/processed/general-chat-sft-v2')
    manifest=json.loads((general/'dataset_manifest.json').read_text())
    mixture=dict(format='codexa-chat-mixture-v1',admitted_for_training=False,
        sources=[dict(path=str(general/'train.jsonl'),sha256=manifest['output_sha256']['train']),
                 dict(path=curated['path'],sha256=curated['sha256'])],
        tokenizer_sha256=TOKENIZER_SHA256,token_probabilities={'ultrachat':.60,'oasst1':.25,'curated':.15},
        maximum_source_passes=2,packing=False,review_required=True,
        curated_token_ceiling=curated['supervised_tokens']*2,
        proposed_total_token_ceiling=curated['supervised_tokens']*2/.15,
        status='PROPOSED_NOT_ADMITTED',
        limitations=['No human factual/open-ended review is represented as completed.',
                     'A quota window ends at source exhaustion; no silent replacement or quota changes.'])
    atomic_document(directory/'mixture.proposed.json',mixture)
    low_exposure={**mixture,'token_probabilities':{'ultrachat':.60,'oasst1':.395,'curated':.005},
        'proposed_total_token_ceiling':curated['supervised_tokens']*2/.005,
        'rationale':'Measured tiny curated pool rules out sustained 15% exposure; 0.5% is a bounded pilot hypothesis.'}
    atomic_document(directory/'mixture-low-exposure.proposed.json',low_exposure)
    metadata=protected_metadata(root)
    report=dict(format='codexa-pretraining-readiness-v1',resources=resources,
        protected_metadata=metadata,curated=curated,environment=experiment_environment(root),
        model=asdict(load_config(root/'configs/1b.yaml').model),
        initializer=None,full_model_gpu_qualified=False,
        suite_review_complete=False,data_admitted=False,
        no_gpu_job_launched=True,no_checkpoint_removed=True)
    atomic_document(directory/'preparation.json',report)
    write_proposals(root,directory,report)
    return report


def write_proposals(root: Path, directory: Path, report: dict) -> None:
    """Freeze proposed hyperparameters while leaving unverified decisions blocked."""
    project=load_config(root/'configs/1b.yaml')
    general=resolve_input(root,'data/processed/general-chat-sft-v2')
    hashes=json.loads((general/'dataset_manifest.json').read_text())['output_sha256']
    validation_indices=stratified_validation_indices(general/'validation.jsonl')
    for name,seconds,max_steps in [('control',3600,512),('improved',3600,512),('10h',36000,5000)]:
        training=asdict(project.training)
        training.update(micro_batch_size=1,gradient_accumulation_steps=8,learning_rate=5e-6,
            weight_decay=.1,warmup_steps=20,max_steps=max_steps,gradient_clip=1,precision='bf16',
            checkpoint_interval=128,evaluation_interval=64,seed=42)
        data=dict(train=str(general/'train.jsonl'),validation=str(general/'validation.jsonl'),
            train_sha256=hashes['train'],validation_sha256=hashes['validation'],
            manifest_sha256=file_sha256(general/'dataset_manifest.json'),maximum_epochs=1,
            workers=0,bucket_size=128,balanced=False,packing=False,
            sequence_policy='complete_conversation_dynamic_right_padding')
        data['validation_indices']=validation_indices
        data['validation_exposure']='potentially historically exposed; CE is a development diagnostic'
        if name=='improved':
            data.update(train=str(directory/'mixture-low-exposure.proposed.json'),
                train_sha256=file_sha256(directory/'mixture-low-exposure.proposed.json'),mixture_manifest=True,
                manifest_sha256=file_sha256(directory/'mixture-low-exposure.proposed.json'),maximum_epochs=2,
                token_probabilities={'ultrachat':.60,'oasst1':.395,'curated':.005})
        if name=='10h':
            data.update(token_probabilities={'ultrachat':.60,'oasst1':.40},
                sampling='finite_token_quota_window_no_replacement',maximum_epochs=1)
        config=dict(format='codexa-readiness-proposal-v1',status='BLOCKED_NOT_AUTHORIZED',
            run_name=f'codexa-934m-chat-readiness-{name}-v1',
            mandatory_gates=dict(initializer_selected=False,data_review_complete=False,
                evaluation_review_complete=False,measured_gpu_shape=False,explicit_approval=False,
                pilot_accepted=False if name=='10h' else None),
            training_operator='scripts/chat_training.py; seal existing launch-manifest before start',
            spec=dict(purpose='production' if name=='10h' else 'experimental',model=asdict(project.model),
                initialization=None,training=training,tokenizer=dict(path=str(resolve_input(root,
                    'checkpoints/tokenizer-base-v1/tokenizer.json')),sha256=TOKENIZER_SHA256),data=data,
                optimization=dict(optimizer='adamw8bit',betas=[.9,.95],epsilon=1e-8,
                    gradient_checkpointing=True,minimum_lr_ratio=.5,tf32=False,compile=False,
                    deterministic_algorithms=True,cublas_workspace_config=':4096:8',
                    loss_normalization='supervised_tokens',target_token_weights={},minimum_headroom_bytes=1536*1024**2),
                evaluation=dict(development_suite_path=str(directory/'development-suite.json'),
                    development_suite_sha256=suite_sha256(development_suite()),
                    acceptance_suite_path=str(directory/'acceptance-suite.sealed.json'),
                    acceptance_suite_sha256=suite_sha256(acceptance_suite()),
                    fresh_behavioral_baseline=True,validation_batches=64,quality_interval=128,
                    source_stratified_losses=True,
                    quality_reserve_seconds=600,validation_reserve_seconds=120,regression_patience=2,
                    regression_policy=dict(task_drop=.05,instruction_drop=.05,factual_drop=.05,
                        correct_END_drop=.10,END_drop=.10,degeneration_increase=.10,empty_increase=.05,
                        role_switch_increase=.05,special_token_maximum=.25,length_ratio_minimum=.30)),
                checkpoint=dict(storage_backend='legacy',slots=6,reserve_bytes=100*1024**3,
                    reclamation_mode='disabled',minimum_save_seconds=120),
                wall_clock_seconds=seconds,finalization_reserve_seconds=600,
                minimum_disk_free_bytes=180*1024**3,
                assistant_token_budget=(int(min(2000000,report['curated']['supervised_tokens']*2/.005))
                                        if name!='10h' else None),
                architecture_sha256=file_sha256(root/'configs/1b.yaml')),
            acceptance=dict(instruction_gain_minimum=.10,factual_gain_minimum=.05,
                paired_objective_ci_lower_bound=0,severe_degeneration_increase_maximum=.02,
                retention_regression_allowed=False,human_review_required=True),
            unresolved=['Initializer awaits approved common-suite generation and human review.',
                'Pilot token ceiling is capacity-derived; achievable cost/cadence/final-save reserve await realistic measurements.',
                'Ten-hour mixture must be selected from pilot evidence; tiny curated pool cannot be cycled indefinitely.'])
        (directory/f'chat-{name}.proposed.yaml').write_text(yaml.safe_dump(config,sort_keys=False))


def baseline_plan(root: Path, directory: Path) -> dict:
    """Bind a future inference-only job to exact protected inputs and suite."""
    suite=json.loads((directory/'development-suite.json').read_text())
    plan=dict(format='codexa-inference-baseline-plan-v1',suite_sha256=suite_sha256(suite),
        candidates={name:dict(path=str(path),sha256=file_sha256(path))
                    for name,path in candidate_paths(root).items()},
        output=str(directory/'baseline-generations'),max_wall_clock_seconds=3600,
        decoding=dict(max_new_tokens=128,do_sample=False,repetition_penalty=1,seed=42),
        script_sha256=file_sha256(root/'scripts/conversational_readiness.py'),
        implementation_sha256=file_sha256(Path(__file__)),
        scoring_sha256=file_sha256(Path(__file__).with_name('behavioral_readiness.py')),
        evaluator_sha256=file_sha256(Path(__file__).with_name('chat_evaluation.py')),
        resources=resource_preflight(root),no_training_updates=True)
    validation=resolve_input(root,'data/processed/general-chat-sft-v2/validation.jsonl')
    plan['validation']=dict(path=str(validation),sha256=file_sha256(validation),
        indices=stratified_validation_indices(validation),source_stratified=True,
        provenance='development loss only; historical exposure not excluded')
    # Resources are rechecked at execution, not part of the immutable job identity.
    identity={k:v for k,v in plan.items() if k!='resources'}
    plan['job_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return plan


def validate_baseline_approval(plan: dict, approval: Path) -> None:
    """Require a precise job grant and live resource gates before creating outputs."""
    authorization=json.loads(approval.read_text())
    if (authorization.get('format')!='codexa-inference-baseline-approval-v1'
            or authorization.get('approved') is not True
            or authorization.get('job_sha256')!=plan['job_sha256']):
        raise ValueError('Separate inference-only approval does not match this job.')
    if not all(plan['resources'][key] for key in ('disk_gate_passed','host_gate_passed','gpu_gate_passed')):
        raise RuntimeError('Inference resource gate failed; no process terminated.')


def run_baseline_candidate(root: Path, directory: Path, approval: Path, name: str, *, started: float) -> dict:
    """An approved child loads one model, then exits to release process-owned memory."""
    plan=json.loads((directory/'active-baseline-plan.json').read_text())
    identity={k:v for k,v in plan.items() if k not in ('resources','job_sha256')}
    if hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=plan['job_sha256']:
        raise ValueError('Active comparison plan changed.')
    for key,path in [('script_sha256',root/'scripts/conversational_readiness.py'),
                     ('implementation_sha256',Path(__file__)),
                     ('scoring_sha256',Path(__file__).with_name('behavioral_readiness.py')),
                     ('evaluator_sha256',Path(__file__).with_name('chat_evaluation.py'))]:
        if file_sha256(path)!=plan[key]:
            raise ValueError('Approved inference implementation changed.')
    plan['resources']=resource_preflight(root)
    validate_baseline_approval(plan,approval)
    authorization=json.loads(approval.read_text())
    if 'original_budget' in authorization:
        deadline=authorization['original_budget']['deadline_monotonic']
        if started+3600>deadline+.01:
            raise ValueError('Candidate cannot reset the original comparison deadline.')
    if time.monotonic()>=started+3600:
        raise TimeoutError('Original comparison deadline expired.')
    if suite_sha256(json.loads((directory/'development-suite.json').read_text()))!=plan['suite_sha256']:
        raise ValueError('Development suite drift.')
    record=plan['candidates'][name]
    if file_sha256(Path(record['path']))!=record['sha256']:
        raise ValueError('Candidate identity changed before load.')
    from .chat_evaluation import evaluate_candidate
    from functools import partial
    from torch.utils.data import DataLoader,Subset
    from llm_data.chat_dataset import IndexedChatDataset,collate_chat
    tokenizer=resolve_input(root,'checkpoints/tokenizer-base-v1/tokenizer.json')
    data=IndexedChatDataset(Path(plan['validation']['path']),load_tokenizer(tokenizer))
    if data.sha256!=plan['validation']['sha256']:
        raise ValueError('Validation identity drift.')
    loader=DataLoader(Subset(data,plan['validation']['indices']),batch_size=1,
        collate_fn=partial(collate_chat),pin_memory=True)
    result=evaluate_candidate(Path(record['path']),tokenizer,Path(plan['output'])/name,
        cases=json.loads((directory/'development-suite.json').read_text()),deadline=started+3600,
        validation_loader=loader,validation_identity=plan['validation'])
    if result['checkpoint_sha256']!=record['sha256'] or result['common_validation']['loss'] is None:
        raise ValueError('Candidate identity or complete validation failed.')
    return result


def run_baselines(root: Path, directory: Path, approval: Path, *, started: float | None = None,
                  resume_incomplete: bool = False) -> dict:
    """Execute only the precisely approved inference job; never training or acceptance."""
    started=time.monotonic() if started is None else started
    plan=baseline_plan(root,directory)
    validate_baseline_approval(plan,approval)
    output=Path(plan['output'])
    if not resume_incomplete:
        output.mkdir(exist_ok=False)
    atomic_document(directory/'active-baseline-plan.json',plan)
    results={}
    if resume_incomplete:
        prior=json.loads((output/'completed-evidence.json').read_text())
        if any(prior['plan'][key]!=plan[key] for key in
               ('candidates','suite_sha256','decoding','validation','scoring_sha256','evaluator_sha256')):
            raise ValueError('Incomplete comparison cannot change scientific protocol.')
        results=prior['results']
        for name,result in results.items():
            if (result['metrics']['cases']!=240 or result['checkpoint_sha256']!=plan['candidates'][name]['sha256']
                    or result['suite_sha256']!=plan['suite_sha256'] or result['common_validation']['loss'] is None):
                raise ValueError('Prior candidate is incomplete or incompatible.')
    candidate_resources=[]
    try:
        for name,record in plan['candidates'].items():
            if name in results:
                continue
            if time.monotonic()-started>=plan['max_wall_clock_seconds']:
                raise TimeoutError('Baseline inference budget exhausted; no partial aggregate scored.')
            live=resource_preflight(root)
            candidate_resources.append(dict(candidate=name,resources=live))
            atomic_document(output/'candidate-resources.json',candidate_resources)
            if not all(live[key] for key in ('disk_gate_passed','host_gate_passed','gpu_gate_passed')):
                raise RuntimeError('Resource gate failed before candidate '+name)
            from .chat_operations import watch_owned_process
            from llm_training.session import SessionBudget
            budget=SessionBudget.start(3600,120);budget.deadline_monotonic=started+3600
            budget.started_utc=time.time()-(time.monotonic()-started);budget.deadline_utc=budget.started_utc+3600
            command=[sys.executable,str(root/'scripts/conversational_readiness.py'),'baseline-candidate',
                '--output',str(directory),'--approval-file',str(approval),'--candidate',name,
                '--original-started-monotonic',str(started)]
            control=directory/f'candidate-{name}.supervisor';control.mkdir(exist_ok=False)
            with (control/'console.log').open('x') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                atomic_document(control/'process.json',dict(worker_pid=child.pid,budget=budget.document()))
                try:
                    code=watch_owned_process(child,budget,control)
                finally:
                    if child.poll() is None:
                        import signal,os
                        os.killpg(child.pid,signal.SIGTERM)
                        try:
                            child.wait(timeout=max(.1,min(30,budget.remaining)))
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid,signal.SIGKILL);child.wait()
            if code:
                raise RuntimeError(f'Candidate {name} failed: returncode {code}; see {control}')
            results[name]=json.loads((output/name/'report.json').read_text())
            atomic_document(output/'completed-evidence.json',dict(plan=plan,results=results,
                full_comparison_complete=False,initializer_selection_allowed=False,human_review_complete=False))
        atomic_document(output/'comparison.json',dict(plan=plan,results=results,human_review_complete=False))
        return results
    except BaseException as error:
        atomic_document(output/'completed-evidence.json',dict(plan=plan,results=results,
            full_comparison_complete=False,missing_candidates=sorted(set(plan['candidates'])-set(results)),
            human_review_complete=False,initializer_selection_allowed=False))
        atomic_document(output/'incomplete.json',dict(error=f'{type(error).__name__}: {error}',
            completed_candidates=list(results),initializer_selected=False))
        raise


def proposal_preflight(root: Path, proposal: Path) -> dict:
    """Explain every unresolved proposed-run gate without allocating model tensors."""
    document=yaml.safe_load(proposal.read_text())
    if document.get('format')!='codexa-readiness-proposal-v1':
        raise ValueError('Unknown readiness proposal format.')
    spec=document['spec']
    if (spec['checkpoint']['storage_backend']!='legacy'
            or spec['checkpoint']['reclamation_mode']!='disabled'
            or spec['minimum_disk_free_bytes']<180*1024**3):
        raise ValueError('Proposal cannot change backend or reduce reserve.')
    gates={k:v for k,v in document['mandatory_gates'].items() if v is not None}
    resources=resource_preflight(root)
    gates.update(disk_capacity=resources['disk_gate_passed'],host_capacity=resources['host_gate_passed'],
                 gpu_capacity=resources['gpu_gate_passed'])
    gates['initializer_selected']=spec['initialization'] is not None
    gates['tokenizer_identity']=file_sha256(Path(spec['tokenizer']['path']))==spec['tokenizer']['sha256']
    for split in ('train','validation'):
        gates[split+'_identity']=file_sha256(Path(spec['data'][split]))==spec['data'][split+'_sha256']
    if spec['data'].get('mixture_manifest'):
        gates['data_admitted']=json.loads(Path(spec['data']['train']).read_text()).get('admitted_for_training') is True
    return dict(proposal_sha256=file_sha256(proposal),resources=resources,gates=gates,
        readiness='BLOCKED' if not all(gates.values()) else 'READY_WITH_EXPLICIT_EXECUTION_APPROVAL',
        blockers=[name for name,passed in gates.items() if not passed],no_training_launched=True)


def admit_mixture(manifest: Path, review: Path, destination: Path) -> dict:
    """Create a new admitted manifest only after traceable human review, never edit sources."""
    document=json.loads(manifest.read_text()); decision=json.loads(review.read_text())
    if document.get('format')!='codexa-chat-mixture-v1' or destination.exists():
        raise ValueError('Known mixture and fresh admission destination required.')
    if not destination.resolve().is_relative_to(output_root(workspace_root()).resolve()):
        raise ValueError('Admission must use a new integration output artifact.')
    if not decision.get('approved') or not decision.get('reviewer') or decision.get('method')!='human_manual_review':
        raise ValueError('Independent human data review is incomplete.')
    if decision.get('manifest_sha256')!=file_sha256(manifest):
        raise ValueError('Review is for another proposed mixture.')
    required=set()
    for source in document['sources']:
        path=Path(source['path'])
        if file_sha256(path)!=source['sha256']:
            raise ValueError('Reviewed source identity drift.')
        with path.open() as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('review_status')=='pending_human':
                    required.add(row['conversation_id'])
    if not required <= set(decision.get('approved_record_ids',[])):
        raise ValueError('Missing factual/open-ended training record reviews.')
    admitted={**document,'admitted_for_training':True,'status':'HUMAN_ADMITTED',
              'review_path':str(review.absolute()),'review_sha256':file_sha256(review),
              'reviewed_manifest_sha256':file_sha256(manifest)}
    atomic_document(destination,admitted)
    return admitted


def select_initializer(comparison: Path, review: Path) -> dict:
    """Select only measured common-suite candidates with completed human review."""
    document=json.loads(comparison.read_text()); reviews=json.loads(review.read_text())
    if (reviews.get('comparison_sha256')!=file_sha256(comparison) or not reviews.get('reviewer')
            or reviews.get('method')!='human_manual_review'):
        raise ValueError('Initializer review must identify the exact common comparison.')
    eligible=[]
    for name,result in document['results'].items():
        if result['suite_sha256']!=document['plan']['suite_sha256']:
            raise ValueError('Incomparable initializer suite identity.')
        assessment=reviews.get('checkpoints',{}).get(result['checkpoint_sha256'],{})
        required={row['id'] for row in result['outputs'] if row['category'] in ('retention','factual','multi_turn')}
        if (result['metrics']['cases']==240 and result['metrics']['entry_gate_passed']
                and assessment.get('accepted') is True and assessment.get('retention_acceptable') is True
                and required <= set(assessment.get('reviewed_ids',[]))):
            eligible.append((name,result))
    if not eligible:
        return dict(status='BLOCKED',initializer=None,
                    reason='No measured candidate passes objective and human semantic entry gates.')
    ranked=sorted(eligible,key=lambda pair:(-pair[1]['metrics']['macro_task_accuracy'],
        pair[1]['metrics']['severe_degeneration_rate'],
        -pair[1]['metrics']['category_pass_rates']['factual'],pair[0]))
    name,result=ranked[0]
    return dict(status='PROVISIONAL_SELECTION',candidate=name,
        initialization=dict(path=result['checkpoint'],sha256=result['checkpoint_sha256'],semantics='weights_initialization'),
        comparison_sha256=file_sha256(comparison),review_sha256=file_sha256(review),
        limitations=['Ranking is provisional; small correlated development populations do not prove superiority.',
                     'New-phase optimizer initialization is not exact resume.'])


def materialize_launch(root: Path, proposal: Path, selection: Path, suite_review: Path,
                       qualification: Path, *, admitted_mixture: Path | None = None,
                       pilot_acceptance: Path | None = None) -> dict:
    """Seal evidence into the existing operator format, with execution still blocked."""
    import copy
    document=yaml.safe_load(proposal.read_text()); selected=json.loads(selection.read_text())
    if document.get('format')!='codexa-readiness-proposal-v1' or not selected.get('initialization'):
        raise ValueError('A measured, reviewed initializer selection is required.')
    review=json.loads(suite_review.read_text()); gpu=json.loads(qualification.read_text())
    spec=copy.deepcopy(document['spec'])
    if (review.get('approved') is not True or review.get('method')!='human_manual_review'
            or not review.get('reviewer') or review.get('suite_sha256')!=spec['evaluation']['development_suite_sha256']):
        raise ValueError('Development prompt/reference review is incomplete.')
    if not (gpu.get('state')=='COMPLETED' and gpu.get('full_model_qualified') is True
            and gpu.get('exact_continuation') is True and gpu.get('throughput_gate_passed') is True
            and gpu.get('restore_gate_passed') is True):
        raise ValueError('Actual full-model qualification has not passed.')
    if spec['checkpoint']['storage_backend']!='legacy' or spec['checkpoint']['reclamation_mode']!='disabled':
        raise ValueError('Preparation cannot promote an experimental storage backend.')
    initial=selected['initialization']
    if file_sha256(Path(initial['path']))!=initial['sha256']:
        raise ValueError('Selected initializer identity drift.')
    if spec['data'].get('mixture_manifest'):
        if admitted_mixture is None:
            raise ValueError('Improved-data pilot requires an admitted mixture.')
        mixture=json.loads(admitted_mixture.read_text())
        if mixture.get('status')!='HUMAN_ADMITTED' or mixture.get('admitted_for_training') is not True:
            raise ValueError('Mixture quality admission is incomplete.')
        spec['data'].update(train=str(admitted_mixture.absolute()),train_sha256=file_sha256(admitted_mixture),
                            manifest_sha256=file_sha256(admitted_mixture))
    if spec['purpose']=='production':
        if pilot_acceptance is None or json.loads(pilot_acceptance.read_text()).get('accepted') is not True:
            raise ValueError('Ten-hour proposal requires accepted paired pilot evidence.')
        raise ValueError('Ten-hour materialization additionally requires measured schedule and accepted final mixture; proposal remains blocked.')
    run_name=document['run_name']; preparation=output_root(root)/'preparation'/run_name
    if preparation.exists():
        raise FileExistsError('Materialized preparation needs a new run namespace.')
    general=resolve_input(root,'data/processed/general-chat-sft-v2')
    config=dict(run_name=run_name,architecture='configs/1b.yaml',tokenizer=spec['tokenizer']['path'],
        tokenizer_sha256=TOKENIZER_SHA256,data=str(general),candidates=[str(p) for p in candidate_paths(root).values()],
        wall_clock_seconds=spec['wall_clock_seconds'],preparation_gpu_seconds=0,maximum_optimizer_steps=512,
        maximum_epochs=spec['data']['maximum_epochs'],micro_batch_size=1,gradient_accumulation_steps=8,
        weight_decay=.1,gradient_clip=1.,seed=42,minimum_gpu_headroom_bytes=1536*1024**2,
        minimum_disk_headroom_bytes=100*1024**3,minimum_host_available_bytes=16*1024**3,
        checkpoint_storage_backend='legacy',checkpoint_reclamation_mode='disabled',
        evaluation_suite_version='codexa-behavior-strict-v1')
    preparation.mkdir(parents=True)
    path=preparation/'operator-config.yaml'; path.write_text(yaml.safe_dump(config,sort_keys=False))
    spec['initialization']=initial
    required_gpu=gpu['peak_gpu_reserved_bytes']+1536*1024**2
    manifest=dict(readiness='BLOCKED',mandatory_gates=dict(
        initializer_selected=True,data_review_complete=True,evaluation_review_complete=True,
        full_model_qualification=True,explicit_execution_approval=False),spec=spec,
        configuration_file=str(path),configuration_file_sha256=file_sha256(path),
        environment=experiment_environment(root),required_free_gpu_bytes=required_gpu,
        evidence_sha256={str(p):file_sha256(p) for p in (proposal,selection,suite_review,qualification)},
        explicit_approval_required=True)
    atomic_document(preparation/'launch-manifest.json',manifest)
    return dict(configuration=str(path),preparation=str(preparation),readiness='BLOCKED_APPROVAL_REQUIRED')


def authorize_launch(preparation: Path, approval: Path) -> dict:
    """Record exact bounded-pilot approval; do not spawn a trainer."""
    manifest=json.loads((preparation/'launch-manifest.json').read_text()); decision=json.loads(approval.read_text())
    if (decision.get('format')!='codexa-conversational-pilot-approval-v1' or decision.get('approved') is not True
            or decision.get('configuration_sha256')!=manifest['configuration_file_sha256']
            or decision.get('run_name')!=Path(preparation).name
            or manifest['spec']['wall_clock_seconds']!=3600 or manifest['spec']['training']['max_steps']>512):
        raise ValueError('Separate explicit approval must identify this bounded pilot configuration.')
    for path,digest in manifest['evidence_sha256'].items():
        if file_sha256(Path(path))!=digest:
            raise ValueError('Sealed launch evidence changed.')
    manifest['mandatory_gates']['explicit_execution_approval']=True
    manifest['readiness']='READY'
    manifest['execution_approval']=decision
    atomic_document(preparation/'launch-manifest.json',manifest)
    return dict(readiness='READY_WITH_FRESH_RESOURCE_PREFLIGHT',no_training_launched=True)
