"""Explicitly approved, bounded 934M qualification. Never a production launcher."""

from dataclasses import replace
import hashlib
import json
import os
import resource
from pathlib import Path
import shutil
import statistics
import time
import uuid
import warnings
from unittest.mock import patch

import numpy as np
import torch

from llm_architecture.model import LanguageModel, ModelConfig, count_parameters
from llm_architecture.checkpointing import read_native_checkpoint
from llm_architecture.component_checkpoint import atomic_json
from llm_training.config import ProjectConfig, TrainingConfig
from llm_training.checkpointing import build_checkpoint_payload, load_checkpoint
from llm_training.recovery import RecoveryStore
from llm_training.training import create_adamw_optimizer, cosine_learning_rate
from llm_training.session import StopRequest
from llm_training.conversational_training import start_chat_viewer
from .chat_preparation import hardware


def qualification_plan(checkpoint: Path, output: Path, *, updates: int = 8,
                       sequence_length: int = 2048) -> dict:
    """Observe resources without constructing a model or creating output directories."""
    if not 2 <= updates <= 8 or not 64 <= sequence_length <= 2048:
        raise ValueError('Qualification is bounded to 2–8 updates and 64–2048 tokens.')
    checkpoint, output = Path(checkpoint).absolute(), Path(output).absolute()
    parent = output
    while not parent.exists():
        parent = parent.parent
    size = checkpoint.stat().st_size
    # Two native, three optimized states (including mid-run), staging and logs.
    peak = 6 * size + 1024**3
    required = max(180*1024**3,100*1024**3+peak)
    observed = hardware()
    free = shutil.disk_usage(parent).free
    return dict(checkpoint=str(checkpoint), output=str(output), updates=updates,
        sequence_length=sequence_length, synthetic_tokens=True, precision='bf16',
        optimizer='adamw8bit', gradient_checkpointing=True,
        expected_peak_storage_bytes=peak, required_free_bytes=required,
        actual_free_bytes=free, storage_gate_passed=free >= required,
        hardware=observed, execution_requires_separate_approval=True,
        max_wall_clock_seconds=900, checkpoint_reclamation_enabled=False,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        wrapper_sha256=hashlib.sha256((Path(__file__).parents[2]/'scripts/storage/checkpoint_gpu_qualification.py').read_bytes()).hexdigest(),
        full_model_qualified=False)


def _equal(a, b, path='state') -> None:
    if isinstance(a, torch.Tensor):
        if (a.dtype != b.dtype or a.shape != b.shape
                or not torch.equal(a.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                                   b.detach().cpu().contiguous().reshape(-1).view(torch.uint8))):
            raise ValueError('Non-bitwise tensor restoration: '+path)
    elif isinstance(a,np.ndarray):
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            raise ValueError('Array restoration differs: '+path)
    elif isinstance(a,dict):
        if a.keys() != b.keys():
            raise ValueError('State keys differ: '+path)
        for key in a:
            _equal(a[key],b[key],path+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('State sequence differs: '+path)
        for index,(x,y) in enumerate(zip(a,b)):
            _equal(x,y,path+'.'+str(index))
    elif a != b:
        raise ValueError('State metadata differs: '+path)


def qualification_execution_preflight(plan: dict, approval: Path) -> dict:
    """Validate a precise authorization and live capacity without constructing a model."""
    from llm_architecture.checkpoint_inspection import inspect_checkpoint_storage
    if not plan['storage_gate_passed']:
        raise RuntimeError('Production storage reserve failed; full-model execution refused.')
    authorization = json.loads(Path(approval).read_text())
    if (authorization.get('format') != 'codexa-gpu-qualification-approval-v1'
            or authorization.get('approved') is not True
            or authorization.get('output') != plan['output']
            or authorization.get('updates') != plan['updates']
            or authorization.get('sequence_length') != plan['sequence_length']
            or authorization.get('implementation_sha256') != plan['implementation_sha256']
            or authorization.get('wrapper_sha256') != plan['wrapper_sha256']
            or authorization.get('max_wall_clock_seconds') != plan['max_wall_clock_seconds']):
        raise ValueError('Separate GPU qualification approval does not match this bounded job.')
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting the process.')
    evidence = inspect_checkpoint_storage(Path(plan['checkpoint']))
    if authorization.get('checkpoint_sha256') != evidence['sha256']:
        raise ValueError('Approved checkpoint fingerprint differs.')
    model_bytes = evidence['components']['model_state_dict']['bytes']
    optimizer_bytes = evidence['components']['optimizer_state_dict']['bytes']
    if not evidence['components']['optimizer_state_dict']['by_dtype'].get('torch.uint8'):
        raise ValueError('This bounded harness requires verified 8-bit optimizer state.')
    if not evidence['complete_optimizer_state']:
        raise ValueError('Qualification requires complete optimizer recovery state.')
    live = hardware()
    gpu_need = 2*model_bytes+optimizer_bytes+int(2.5*1024**3)
    if (live['free_gpu_bytes'] < gpu_need
            or live['host_memory_bytes']['MemAvailable'] < 16*1024**3):
        raise RuntimeError('Live GPU/RAM headroom does not fit bounded qualification; no process changed.')
    output = Path(plan['output'])
    if output.exists():
        raise FileExistsError('Qualification requires a fresh output namespace.')
    if shutil.disk_usage(output.parent).free < plan['required_free_bytes']:
        raise RuntimeError('Actual storage capacity changed; qualification refused.')
    return evidence


def execute_qualification(plan: dict, approval: Path, *, started: float) -> dict:
    """No execution occurs without approval, passing reserve and live resource checks."""
    evidence = qualification_execution_preflight(plan, approval)
    output = Path(plan['output'])
    output.mkdir()
    report = dict(plan, evidence=evidence, state='STARTING', full_model_qualified=False)
    atomic_json(output/'qualification.json',report)
    def boundary(stage):
        if time.monotonic()-started > 780:
            raise TimeoutError('Qualification finalization reserve reached at '+stage)
    try:
        progress=dict(run_name=output.name,phase='checkpoint qualification',state='STARTING',
                      step=0,total_steps=plan['updates'] + plan['updates']//2,
                      total_tokens_seen=0)
        atomic_json(output/'progress.json',progress)
        viewer=start_chat_viewer(output/'progress.json',title='Codexa 934M checkpoint qualification')
        boundary('load')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        project=ProjectConfig(ModelConfig(**evidence['config']['model']),
                              TrainingConfig(**evidence['config']['training']))
        model=LanguageModel(project.model).to('cuda')
        model.set_gradient_checkpointing(True)
        if count_parameters(model)!=934356480:
            raise ValueError('Qualification architecture is not the exact 934M model.')
        optimizer=create_adamw_optimizer(model,learning_rate=project.training.learning_rate,
            weight_decay=project.training.weight_decay,optimizer_name='adamw8bit',
            betas=(.9,.95),epsilon=1e-8)
        restored=load_checkpoint(Path(plan['checkpoint']),model=model,optimizer=optimizer,
                                 scaler=None,expected_config=project)
        initial_state=restored.state
        context=dict(restored.resume_context or {})
        context['qualification']=dict(synthetic_cursor=0,source_sha256=evidence['sha256'])
        identity=uuid.uuid4().hex
        def payload(state):
            return build_checkpoint_payload(model=model,optimizer=optimizer,scaler=None,
                state=state,scheduler=restored.scheduler,config=project,
                run_name='codexa-934m-v3-qualification',run_id=identity,
                tokenizer_reference=restored.tokenizer_reference,
                tokenizer_sha256=restored.tokenizer_sha256,resume_context=context)
        native=RecoveryStore(output/'native')
        optimized=RecoveryStore(output/'optimized',backend='optimized',reserve_bytes=100*1024**3,
            retention=dict(continuous=True,max_checkpoint_bytes=int(evidence['file_bytes']*1.1)+16*1024**2))
        initial=payload(initial_state)
        source=read_native_checkpoint(Path(plan['checkpoint']))
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state','random_states',
                    'training_state','grad_scaler_state_dict'):
            _equal(source[key], initial[key], 'native_restore.'+key)
        if restored.resume_context != source.get('resume_context'):
            raise ValueError('Native sampler/provenance metadata was not restored.')
        del source
        times={}
        for name,store in (('native_initial',native),('optimized_initial',optimized)):
            boundary(name)
            before=time.monotonic()
            store.save(initial,best=True)
            times[name]=time.monotonic()-before
        loaded=optimized.load()
        _equal(initial,loaded)
        del initial,loaded
        before=time.monotonic()
        native_path=native.resolve()
        before=time.monotonic()
        load_checkpoint(native_path,model=model,optimizer=optimizer,scaler=None,expected_config=project)
        torch.cuda.synchronize()
        times['native_restore_to_gpu']=time.monotonic()-before
        scheduler = restored.scheduler
        def updates(requested, state, count):
            durations=[]
            model.train()
            for _ in range(count):
                boundary('optimizer-update')
                if requested():
                    raise InterruptedError('Controlled qualification interruption.')
                torch.cuda.synchronize()
                before=time.monotonic()
                tokens=torch.randint(8,project.model.vocab_size,(1,plan['sequence_length']+1))
                inputs=tokens[:,:-1].to('cuda')
                labels=tokens[:,1:].to('cuda')
                labels[:,:plan['sequence_length']//2]=-100
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    _,loss=model(inputs,labels)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite qualification loss.')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),project.training.gradient_clip)
                if not torch.isfinite(norm):
                    raise ValueError('Nonfinite qualification gradient.')
                learning_rate = cosine_learning_rate(state.optimizer_step,
                    warmup_steps=scheduler.warmup_steps, max_steps=scheduler.max_steps,
                    peak_learning_rate=scheduler.peak_learning_rate,
                    minimum_learning_rate=scheduler.minimum_learning_rate)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state=replace(state,optimizer_step=state.optimizer_step+1,
                              micro_step=state.micro_step+1,
                              batches_in_epoch=state.batches_in_epoch+1,
                              tokens_seen=state.tokens_seen+int((labels != -100).sum()))
                context['qualification']['synthetic_cursor'] += 1
                torch.cuda.synchronize()
                durations.append(time.monotonic()-before)
                progress.update(state='RUNNING',step=progress['step']+1,
                    total_tokens_seen=progress['total_tokens_seen']+plan['sequence_length'],
                    training_loss=float(loss),tokens_per_second=plan['sequence_length']/durations[-1],
                    elapsed_seconds=time.monotonic()-started,
                    remaining_seconds=max(0,900-(time.monotonic()-started)),
                    checkpoint_path=str(optimized.store_root/optimized.index()['latest']['path']),
                    eta_seconds=durations[-1]*(progress['total_steps']-progress['step']),
                    viewer_closed=viewer.poll() is not None)
                atomic_json(output/'progress.json',progress)
            return state,durations
        with StopRequest() as requested:
            midpoint = plan['updates'] - plan['updates']//2
            middle,first_durations=updates(requested, initial_state, midpoint)
            boundary('optimized-midpoint')
            before=time.monotonic(); optimized.save(payload(middle))
            times['optimized_midpoint']=time.monotonic()-before
            continuous,durations_a=updates(requested, middle, plan['updates']//2)
            boundary('native-final')
            before=time.monotonic();native.save(payload(continuous),final=True)
            times['native_final']=time.monotonic()-before
            before=time.monotonic()
            resumed_checkpoint=load_checkpoint(None,recovery_store=optimized,model=model,
                optimizer=optimizer,scaler=None,expected_config=project)
            torch.cuda.synchronize()
            times['optimized_restore_to_gpu']=time.monotonic()-before
            context=dict(resumed_checkpoint.resume_context)
            scheduler=resumed_checkpoint.scheduler
            resumed,durations_b=updates(requested, resumed_checkpoint.state, plan['updates']//2)
            before=time.monotonic();optimized.save(payload(resumed),final=True)
            times['optimized_final']=time.monotonic()-before
        expected=read_native_checkpoint(native.resolve())
        actual=optimized.load()
        for key in ('model_state_dict','optimizer_state_dict','scheduler_state','random_states',
                    'grad_scaler_state_dict','training_state','resume_context'):
            _equal(expected[key],actual[key],key)
        ratio=statistics.median(durations_b[1:] or durations_b)/statistics.median(durations_a[1:] or durations_a)
        restore_ratio=times['optimized_restore_to_gpu']/times['native_restore_to_gpu']
        import llm_architecture.component_checkpoint as component
        original_manifest=component.manifest_document
        latest=optimized._record_path()
        def unavailable(path, **kwargs):
            if Path(path)==latest:
                raise OSError('Injected unavailable latest; no file changed.')
            return original_manifest(path,**kwargs)
        with patch.object(component,'manifest_document',unavailable), warnings.catch_warnings(record=True) as notices:
            load_checkpoint(None,recovery_store=optimized,allow_previous=True,
                model=model,optimizer=optimizer,scaler=None,expected_config=project)
        fallback=optimized.last_recovery_selection
        if not fallback['fallback'] or not notices:
            raise ValueError('Explicit previous-state fallback was not visible.')
        report.update(state='COMPLETED',exact_continuation=True,checkpoint_times=times,
            native_update_seconds=durations_a,optimized_update_seconds=durations_b,
            initial_update_seconds=first_durations, resumed_from_optimizer_boundary=True,
            throughput_ratio=ratio,throughput_gate_passed=ratio<=1.05,
            restore_ratio=restore_ratio,restore_gate_passed=restore_ratio<=1.10,
            injected_failure_recovery=fallback,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            supervised_tokens_per_update=plan['sequence_length']-plan['sequence_length']//2,
            context_tokens_per_update=plan['sequence_length'],
            # CPU latency and fault gates are independent; this report cannot promote production.
            full_model_qualified=ratio<=1.05 and restore_ratio<=1.10
                and plan['sequence_length']==2048 and plan['updates']==8,
            production_ready=False)
    except BaseException as error:
        report.update(state='FAILED',error=f'{type(error).__name__}: {error}',full_model_qualified=False)
        raise
    finally:
        report['peak_host_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        report['elapsed_seconds']=time.monotonic()-started
        atomic_json(output/'qualification.json',report)
        if 'progress' in locals():
            progress.update(state=report['state'],elapsed_seconds=report['elapsed_seconds'])
            atomic_json(output/'progress.json',progress)
    return report
