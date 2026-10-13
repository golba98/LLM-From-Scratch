"""Verified, visible, hard-bounded conversational SFT pilot; never overwrites old runs."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_training.checkpointing import (CheckpointManager, SchedulerState, build_checkpoint_payload, file_sha256, verify_checkpoint_checksum)
from llm_training.config import load_config
from llm_training.conversational_training import ChatDataset, atomic_json, start_chat_viewer, view_chat
from llm_architecture.model import LanguageModel
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.training import (JsonlRunLogger, TrainingState, autocast_context,
                          evaluate, resolve_precision, set_deterministic_seed, train_model)


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    if '--help' not in sys.argv:
        from training_environment_guard import require_environment
        require_environment('new')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', choices=('development', 'historical'),
                        help='Required training identity; verified before model or optimizer setup')
    parser.add_argument('--view', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=Path('checkpoints/codexa-900m-base-v1/latest.pt'))
    parser.add_argument('--tokenizer', type=Path, default=Path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/general-chat-sft-v2'))
    parser.add_argument('--run-name', default='codexa-900m-chat-stage2-pilot-v1')
    parser.add_argument('--max-steps', type=int, default=200)
    parser.add_argument('--minutes', type=float, default=60)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.view:
        view_chat(args.view)
        return
    if not args.preflight_only and args.minutes <= 5:
        raise ValueError('Reserve five minutes for evaluation; training requires a larger budget.')
    if not 1 <= args.max_steps <= 200 or not 0 < args.minutes <= 60:
        raise ValueError('Pilot must be <=200 steps and <=60 minutes.')
    if Path(args.run_name).name != args.run_name or not args.run_name:
        raise ValueError('Run name must be a safe directory name.')
    log = Path('logs') / args.run_name
    checkpoints = Path('checkpoints') / args.run_name
    if log.exists() or checkpoints.exists():
        raise FileExistsError('Fresh log and checkpoint namespaces are required.')
    log.mkdir(parents=True)
    started = time.time()
    deadline = time.monotonic() + args.minutes * 60
    end_utc = started + args.minutes * 60
    def expired(*_):
        raise TimeoutError('Total Stage 2 pilot budget expired.')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, args.minutes * 60)
    status = log / 'progress.json'
    progress = dict(state='STARTING', step=0, total_steps=args.max_steps, pid=os.getpid(), kind='934M decoder',
                    checkpoint_path=str(checkpoints / 'latest.pt'), total_tokens_seen=0)
    atomic_json(status, progress)
    report = dict(started_utc=started, deadline_utc=end_utc, source=str(args.checkpoint),
                  maximum_steps=args.max_steps, minutes=args.minutes, completed=False)
    viewer = None
    logger = None
    try:
        viewer = start_chat_viewer(status)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('The bounded pilot requires native CUDA BF16.')
        import bitsandbytes as bnb
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader,nounits'], text=True)
        if any(int(pid.strip()) != os.getpid() for pid in processes.splitlines() if pid.strip()):
            raise RuntimeError('Another CUDA compute process is active; leave it untouched and stop this pilot.')
        report['source_sha256'] = verify_checkpoint_checksum(args.checkpoint)
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
        tokenizer = load_tokenizer(args.tokenizer)
        tokenizer_hash = file_sha256(args.tokenizer)
        if payload.get('tokenizer_sha256') != tokenizer_hash:
            raise ValueError('Pilot source must have a verified tokenizer fingerprint.')
        config = load_config('configs/1b.yaml')
        config = replace(config, training=replace(config.training, learning_rate=1e-5, max_steps=args.max_steps,
                                                  warmup_steps=min(20, args.max_steps - 1), checkpoint_interval=50,
                                                  evaluation_interval=50))
        if payload['config']['model'] != asdict(config.model) or tokenizer.get_vocab_size() != config.model.vocab_size:
            raise ValueError('Pilot source architecture/tokenizer mismatch.')
        manifest = json.loads((args.data / 'dataset_manifest.json').read_text())
        for split, digest in manifest['output_sha256'].items():
            if file_sha256(args.data / f'{split}.jsonl') != digest:
                raise ValueError('Conversational dataset checksum mismatch.')
        set_deterministic_seed(42)
        training = ChatDataset(args.data / 'train.jsonl', tokenizer, context=2048, limit=20000, seed=42)
        validation = ChatDataset(args.data / 'validation.jsonl', tokenizer, context=2048, limit=256, seed=43)
        test = ChatDataset(args.data / 'test.jsonl', tokenizer, context=2048, limit=256, seed=44)
        if training.groups & validation.groups or training.groups & test.groups or validation.groups & test.groups:
            raise ValueError('Dataset groups leak across splits.')
        report['data'] = dict(train=len(training), validation=len(validation), test=len(test),
                              sources=training.sources, manifest=manifest)
        device = torch.device('cuda')
        precision = resolve_precision('bf16', device)
        model = LanguageModel(config.model).to(device)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        model.set_gradient_checkpointing(True)
        train_loader = DataLoader(training, batch_size=1, shuffle=True, generator=torch.Generator().manual_seed(42))
        validation_loader = DataLoader(validation, batch_size=1)
        test_loader = DataLoader(test, batch_size=1)
        report['baseline_validation_loss'] = evaluate(model, validation_loader, device=device, precision=precision, max_batches=32)[0]
        progress.update(state='PREFLIGHT', validation_loss=report['baseline_validation_loss'])
        atomic_json(status, progress)
        if viewer.poll() is not None:
            raise RuntimeError('Required viewer closed before preflight update.')
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, weight_decay=0.1)
        inputs, targets = next(iter(train_loader))
        model.train()
        torch.cuda.reset_peak_memory_stats()
        before = model.final_norm.weight.detach().clone()
        # A second update measures steady-state activations with optimizer states allocated.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, precision):
                _, loss = model(inputs.to(device), targets.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite preflight loss.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        changed = not torch.equal(before, model.final_norm.weight.detach())
        report['preflight'] = dict(loss=float(loss.detach()), gradient_norm=float(norm), gradient_update=changed,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved())
        if not changed or torch.cuda.max_memory_reserved() >= 14 * 1024**3:
            raise RuntimeError('Preflight update or 14 GiB memory gate failed.')
        atomic_json(log / 'preflight.json', report)
        # Preflight never becomes a training initialization: restore exact base weights.
        del optimizer, loss, inputs, targets, before
        model.zero_grad(set_to_none=True)
        model.load_state_dict(payload['model_state_dict'], strict=True)
        del payload
        gc.collect()
        torch.cuda.empty_cache()
        set_deterministic_seed(42)
        if args.preflight_only:
            report['state'] = 'PREFLIGHT_ONLY'
        else:
            optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=1e-5, weight_decay=0.1)
            logger = JsonlRunLogger(Path('logs'), args.run_name, overwrite=False)
            logger.write_metadata(report)
            manager = CheckpointManager(Path('checkpoints'), args.run_name)
            state = TrainingState(best_validation_loss=report['baseline_validation_loss'])
            run_id = uuid.uuid4().hex
            scheduler = SchedulerState(warmup_steps=config.training.warmup_steps, max_steps=args.max_steps,
                                       peak_learning_rate=1e-5, minimum_learning_rate=1e-6)
            def save(best=False):
                checkpoint = build_checkpoint_payload(model=model, optimizer=optimizer, scaler=None, state=state,
                    scheduler=scheduler, config=config, run_name=args.run_name, run_id=run_id,
                    tokenizer_reference=str(args.tokenizer), tokenizer_sha256=tokenizer_hash,
                    resume_context={'dataset_manifest': manifest, 'deadline_utc': end_utc})
                manager.save(checkpoint, is_best=best)
            best = report['baseline_validation_loss']
            def progress_callback(current, metrics):
                nonlocal best
                if viewer.poll() is not None:
                    raise RuntimeError('Required Kitty viewer closed during training.')
                if metrics.peak_reserved_vram_bytes >= 14 * 1024**3:
                    raise RuntimeError('Runtime reserved VRAM exceeded the 14 GiB gate.')
                progress.update(state='RUNNING', step=current.optimizer_step, training_loss=metrics.training_loss,
                                validation_loss=metrics.validation_loss if metrics.validation_loss is not None else progress.get('validation_loss'),
                                tokens_per_second=metrics.tokens_per_second, total_tokens_seen=current.tokens_seen,
                                eta_seconds=metrics.step_time_seconds * (args.max_steps - current.optimizer_step))
                atomic_json(status, progress)
                improvement = metrics.validation_loss is not None and metrics.validation_loss < best
                if improvement:
                    best = metrics.validation_loss
                if current.optimizer_step % 50 == 0 or current.optimizer_step == args.max_steps:
                    save(improvement)
            try:
                state, _ = train_model(model, train_loader, optimizer, device=device, precision=precision,
                    max_steps=args.max_steps, gradient_accumulation_steps=32, gradient_clip=1.0,
                    warmup_steps=config.training.warmup_steps, peak_learning_rate=1e-5, minimum_learning_rate=1e-6,
                    seed=42, state=state, run_name=args.run_name, run_id=run_id, logger=logger,
                    validation_loader=validation_loader, evaluation_interval=50, max_validation_batches=32,
                    on_optimizer_step=progress_callback, deadline_monotonic=deadline - 300, progress=True)
            except TimeoutError:
                report['training_deadline_reached'] = True
            if not manager.latest_path.exists() or state.optimizer_step % 50:
                save()
            verify_checkpoint_checksum(manager.latest_path)
            report['final_validation_loss'] = evaluate(model, validation_loader, device=device, precision=precision, max_batches=32)[0]
            report['test_loss'] = evaluate(model, test_loader, device=device, precision=precision, max_batches=32)[0]
            report['optimizer_steps'] = state.optimizer_step
            report['assistant_tokens'] = state.tokens_seen
            report['checkpoint'] = str(manager.latest_path)
            report['completed'] = state.optimizer_step == args.max_steps
            del model, optimizer, training, validation, test, train_loader, validation_loader, test_loader
            gc.collect()
            torch.cuda.empty_cache()
            # Evaluate the same 24 memory-enabled scenarios inside the remaining pilot budget.
            if time.monotonic() < deadline - 30:
                remaining = max(1, deadline - time.monotonic())
                child = subprocess.run([sys.executable, 'scripts/evaluation/evaluate_chat_memory.py', '--checkpoint', str(manager.latest_path),
                                        '--output', str(log / 'chat-evaluation'), '--only-memory', '--deadline-utc', str(end_utc - 10)],
                                       timeout=remaining)
                report['chat_evaluation_exit_code'] = child.returncode
            if time.monotonic() < deadline - 30:
                from scripts.inference.export_chat_native import export_native
                report['export'] = export_native(manager.latest_path, args.tokenizer, Path('exports') / args.run_name)
        report['state'] = 'COMPLETED'
        progress['state'] = 'COMPLETED'
    except Exception as error:
        report.update(state='FAILED', error=str(error))
        progress['state'] = 'FAILED'
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        report['elapsed_seconds'] = time.time() - started
        atomic_json(log / 'report.json', report)
        atomic_json(status, progress)
        if logger:
            logger.write_metadata(report)
            logger.close()


if __name__ == '__main__':
    main()
