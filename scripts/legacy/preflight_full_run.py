"""Run reproducible hardware, data, storage, and backup preflight checks."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train import _load_manifest, validate_manifest
from llm_training.config import load_config
from llm_architecture.model import LanguageModel, count_parameters
from llm_training.preflight import (
    PreflightCheck,
    disk_capacity_check,
    estimate_checkpoint_storage,
    independent_filesystems,
    preflight_timestamp,
)


def _gpu_check(maximum_temperature: int) -> tuple[list[PreflightCheck], dict[str, object]]:
    checks: list[PreflightCheck] = []
    details: dict[str, object] = {
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": False,
    }
    if not torch.cuda.is_available():
        checks.append(PreflightCheck("cuda", "fail", "CUDA is unavailable."))
        checks.append(
            PreflightCheck("bf16", "fail", "CUDA BF16 cannot be tested.")
        )
        return checks, details
    details["bf16_supported"] = torch.cuda.is_bf16_supported()
    details["gpu_name"] = torch.cuda.get_device_name(0)
    details["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    checks.append(PreflightCheck("cuda", "pass", str(details["gpu_name"])))
    checks.append(
        PreflightCheck(
            "bf16",
            "pass" if details["bf16_supported"] else "fail",
            f"supported={details['bf16_supported']}",
        )
    )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,power.draw,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        temperature_text, power_text, memory_text = (
            field.strip() for field in result.stdout.strip().split(",")
        )
        temperature = int(temperature_text)
        details.update(
            {
                "temperature_celsius": temperature,
                "power_draw_watts": float(power_text),
                "nvidia_smi_memory_mib": int(memory_text),
            }
        )
        checks.append(
            PreflightCheck(
                "gpu_temperature",
                "pass" if temperature <= maximum_temperature else "fail",
                f"{temperature} C (maximum {maximum_temperature} C)",
            )
        )
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        checks.append(
            PreflightCheck(
                "gpu_temperature",
                "fail",
                f"nvidia-smi query failed: {error}",
            )
        )
    return checks, details


def _power_check(manually_confirmed: bool) -> PreflightCheck:
    supplies = Path("/sys/class/power_supply")
    online_sources: list[str] = []
    if supplies.is_dir():
        for supply in supplies.iterdir():
            try:
                supply_type = supply.joinpath("type").read_text().strip()
                online = supply.joinpath("online").read_text().strip()
            except OSError:
                continue
            if supply_type in {"Mains", "UPS"} and online == "1":
                online_sources.append(f"{supply.name}:{supply_type}")
    if online_sources:
        return PreflightCheck(
            "power_source",
            "pass",
            "online=" + ",".join(online_sources),
        )
    if manually_confirmed:
        return PreflightCheck(
            "power_source",
            "pass",
            "Power stability was explicitly confirmed because sysfs exposed "
            "no Mains/UPS status.",
        )
    return PreflightCheck(
        "power_source",
        "fail",
        "No online Mains/UPS status was exposed by sysfs; "
        "rerun with --confirm-power-stability only after manual confirmation.",
    )


def _backup_checks(
    checkpoint_directory: Path,
    backup_destination: Path | None,
    *,
    required_bytes: int,
    accept_no_independent_backup: bool,
) -> list[PreflightCheck]:
    if backup_destination is not None and accept_no_independent_backup:
        raise ValueError(
            "--backup-destination and --accept-no-independent-backup "
            "cannot be used together."
        )
    if accept_no_independent_backup:
        return [
            PreflightCheck(
                "independent_backup",
                "pass",
                "The operator explicitly accepted running without an "
                "independent checkpoint backup.",
            )
        ]
    if backup_destination is None:
        return [
            PreflightCheck(
                "independent_backup",
                "fail",
                "No backup destination supplied. Provide one or explicitly "
                "accept the risk with --accept-no-independent-backup.",
            )
        ]
    try:
        independent = independent_filesystems(
            checkpoint_directory,
            backup_destination,
        )
        checks = [
            PreflightCheck(
                "independent_backup",
                "pass" if independent else "fail",
                f"destination={backup_destination}",
            )
        ]
        checks.append(
            disk_capacity_check(
                backup_destination,
                required_bytes=required_bytes,
                name="backup_capacity",
            )
        )
        return checks
    except Exception as error:
        return [PreflightCheck("independent_backup", "fail", str(error))]


def _model_smoke(
    model: LanguageModel,
    *,
    sequence_length: int,
) -> PreflightCheck:
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        return PreflightCheck(
            "model_smoke",
            "fail",
            "CUDA BF16 is required for the full-run smoke.",
        )
    device = torch.device("cuda")
    model.to(device)
    model.train()
    input_ids = torch.randint(
        0,
        model.config.vocab_size,
        (1, sequence_length),
        device=device,
    )
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _logits, loss = model(input_ids, input_ids)
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("Model smoke loss is non-finite.")
        loss.backward()
        finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
        if not finite:
            raise FloatingPointError("Model smoke gradients are non-finite.")
        return PreflightCheck(
            "model_smoke",
            "pass",
            f"BF16 forward/backward passed at sequence_length={sequence_length}.",
        )
    except Exception as error:
        return PreflightCheck("model_smoke", "fail", str(error))
    finally:
        model.to("cpu")
        torch.cuda.empty_cache()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_file:
            json.dump(value, output_file, indent=2, sort_keys=True, allow_nan=False)
            output_file.write("\n")
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--token-manifest", type=Path, required=True)
    parser.add_argument("--train-token-file", type=Path, required=True)
    parser.add_argument("--validation-token-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--backup-destination", type=Path)
    parser.add_argument(
        "--accept-no-independent-backup",
        action="store_true",
        help=(
            "Record an explicit operator decision to run without an "
            "independent checkpoint backup."
        ),
    )
    parser.add_argument("--maximum-temperature", type=int, default=70)
    parser.add_argument("--confirm-power-stability", action="store_true")
    parser.add_argument("--model-smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.maximum_temperature <= 0:
        raise ValueError("--maximum-temperature must be positive.")

    project_config = load_config(arguments.config)
    model = LanguageModel(project_config.model)
    parameter_count = count_parameters(model)
    storage = estimate_checkpoint_storage(
        parameter_count=parameter_count,
        max_steps=project_config.training.max_steps,
        checkpoint_interval=project_config.training.checkpoint_interval,
    )
    arguments.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checks: list[PreflightCheck] = [
        PreflightCheck(
            "model_configuration",
            "pass",
            f"parameters={parameter_count}",
        ),
        disk_capacity_check(
            arguments.checkpoint_dir,
            required_bytes=storage.projected_bytes,
        ),
        _power_check(arguments.confirm_power_stability),
    ]
    gpu_checks, hardware = _gpu_check(arguments.maximum_temperature)
    checks.extend(gpu_checks)
    try:
        manifest = _load_manifest(arguments.token_manifest)
        dtype = validate_manifest(
            manifest,
            train_token_file=arguments.train_token_file,
            validation_token_file=arguments.validation_token_file,
            context_length=project_config.model.context_length,
            model_vocab_size=project_config.model.vocab_size,
        )
        checks.append(
            PreflightCheck("token_data", "pass", f"dtype={dtype.name}")
        )
    except Exception as error:
        checks.append(PreflightCheck("token_data", "fail", str(error)))

    checks.extend(
        _backup_checks(
            arguments.checkpoint_dir,
            arguments.backup_destination,
            required_bytes=storage.projected_bytes,
            accept_no_independent_backup=(
                arguments.accept_no_independent_backup
            ),
        )
    )
    if arguments.model_smoke:
        checks.append(
            _model_smoke(
                model,
                sequence_length=min(project_config.model.context_length, 512),
            )
        )
    report = {
        "created_at_utc": preflight_timestamp(),
        "config_path": str(arguments.config),
        "token_manifest_path": str(arguments.token_manifest),
        "model_config": asdict(project_config.model),
        "training_config": asdict(project_config.training),
        "parameter_count": parameter_count,
        "checkpoint_storage_estimate": storage.to_dict(),
        "hardware": hardware,
        "checks": [check.to_dict() for check in checks],
        "passed": not any(check.status == "fail" for check in checks),
        "warnings": sum(check.status == "warning" for check in checks),
    }
    _atomic_json(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
