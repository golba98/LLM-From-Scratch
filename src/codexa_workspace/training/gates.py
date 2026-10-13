"""Production resource gates and read-only host measurements.

The gate values are the single source of truth for every training job; jobs may
only raise them (``validate_gates`` refuses lower or missing values).
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess

from codexa_workspace.paths import REPO_ROOT


GIB = 1024**3
DISK_FLOOR_BYTES = 180 * GIB
RECOVERY_RESERVE_BYTES = 100 * GIB
HOST_RAM_BYTES = 16 * GIB
VRAM_FREE_MIB = 13928
IN_RUN_FREE_VRAM_BYTES = 1536 * 1024**2


class TrainingProhibited(RuntimeError):
    """An optimizer, backward pass, weight update or checkpoint write was attempted outside train mode."""


def canonical_gates() -> dict:
    """The production gates; never redefine or lower them elsewhere."""
    return dict(startup_free_vram_mib=VRAM_FREE_MIB, available_ram_bytes=HOST_RAM_BYTES,
                free_disk_bytes=DISK_FLOOR_BYTES, recovery_reserve_bytes=RECOVERY_RESERVE_BYTES,
                in_run_free_vram_bytes=IN_RUN_FREE_VRAM_BYTES)


def validate_gates(gates: dict) -> dict:
    canonical = canonical_gates()
    weakened = {k: (gates.get(k), v) for k, v in canonical.items() if gates.get(k) is None or gates[k] < v}
    if weakened:
        raise ValueError(f'Resource gates may not be lowered or omitted: {weakened}')
    return gates


def measure_resources(path: Path = REPO_ROOT) -> dict:
    """Read-only host measurements (nvidia-smi, /proc/meminfo, statvfs)."""
    vram = None
    try:
        done = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                               '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
        name, total, used, free = [x.strip() for x in done.stdout.strip().splitlines()[0].split(',')]
        vram = dict(name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free))
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        vram = None
    available = None
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            available = int(line.split()[1]) * 1024
    return dict(vram=vram, available_ram_bytes=available, free_disk_bytes=shutil.disk_usage(path).free,
                measured_utc=datetime.now(timezone.utc).isoformat())


def resource_checks(gates: dict, measured: dict, *, required_disk_bytes: int | None = None) -> list[dict]:
    gates = validate_gates(gates)
    vram = measured.get('vram') or {}
    disk_needed = max(gates['free_disk_bytes'], gates['recovery_reserve_bytes'] + (required_disk_bytes or 0))
    return [
        dict(name='startup_free_vram', passed=vram.get('free_mib', -1) >= gates['startup_free_vram_mib'],
             measured=vram.get('free_mib'), required=gates['startup_free_vram_mib'], unit='MiB'),
        dict(name='available_ram', passed=(measured.get('available_ram_bytes') or -1) >= gates['available_ram_bytes'],
             measured=measured.get('available_ram_bytes'), required=gates['available_ram_bytes'], unit='bytes'),
        dict(name='free_disk_and_recovery_reserve', passed=measured['free_disk_bytes'] >= disk_needed,
             measured=measured['free_disk_bytes'], required=disk_needed, unit='bytes')]


@contextmanager
def no_training_guard(*models):
    """Make optimizer construction, backward passes and torch.save raise; freeze parameters."""
    import torch
    patches = []

    def refuse(what):
        def raiser(*args, **kwargs):
            raise TrainingProhibited(f'{what} is prohibited outside authorized train mode.')
        return raiser

    targets = [(torch.optim.Optimizer, '__init__', 'Optimizer construction'),
               (torch.Tensor, 'backward', 'Backpropagation'),
               (torch.autograd, 'backward', 'Backpropagation'),
               (torch, 'save', 'Checkpoint write')]
    try:
        import bitsandbytes.optim as bnb_optim  # noqa: F401
        for name in ('AdamW8bit', 'Adam8bit', 'PagedAdamW8bit'):
            if hasattr(bnb_optim, name):
                targets.append((getattr(bnb_optim, name), '__init__', f'bitsandbytes {name} construction'))
    except Exception:
        pass
    for owner, name, what in targets:
        patches.append((owner, name, owner.__dict__.get(name, getattr(owner, name))))
        setattr(owner, name, refuse(what))
    for model in models:
        model.requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise TrainingProhibited('Parameters must be frozen during preparation and evaluation.')
    try:
        yield
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)


def host_available_bytes() -> int:
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable not reported.')


def gpu_snapshot() -> dict:
    query = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    if query.returncode:
        return dict(available=False, error=query.stderr.strip()[-300:])
    name, total, used, free = [value.strip() for value in query.stdout.strip().splitlines()[0].split(',')]
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                           '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=30)
    return dict(available=True, name=name, total_mib=int(total), used_mib=int(used), free_mib=int(free),
                compute_apps=[line.strip() for line in apps.stdout.splitlines() if line.strip()])
