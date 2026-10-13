#!/usr/bin/env python3
"""Read-only before/after validation of the masked KV-cache position repair.

Loads the verified baseline and final 934M checkpoints for inference only (no
optimizer, no backward pass, no checkpoint writes) and records:

* the masked-cache logit differential originally measured by
  ``post_training_generation_diagnostic.cache_checks`` (FP32, 8-token prefix);
* a left-padded batched cached decode step against compact references;
* SHA-256 digests of unmasked full and cached logits (bit-identity check);
* the 48-prompt historical greedy benchmark under its original runtime.

Run ``--phase before`` on the unrepaired code, ``--phase after`` on the
repaired code, then ``--phase compare``. Use the integration virtualenv.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# --historical-context keeps the installed .venv wheels (the environment that
# produced codexa-934m-foundation-5h-v1); otherwise source components load first.
HISTORICAL_CONTEXT = "--historical-context" in sys.argv
if HISTORICAL_CONTEXT:
    sys.path.insert(0, str(ROOT / "src"))
else:
    sys.path.insert(0, str(ROOT))
    import workspace_bootstrap  # noqa: E402,F401  (source component paths first)

RUN = ROOT / "outputs/training/codexa-934m-foundation-5h-v1"
DIAGNOSTIC = ROOT / "outputs/diagnostics/post-training-generation-investigation/20261012T040042Z-10a1d256"
ALLOWED = ROOT / "outputs/diagnostics/masked-kv-cache-repair"
FINAL_SHA256 = "ca779a1640fa8f664356faf6db0897c9e9669b90a2e3ed35a7a23fc13144236e"

_spec = importlib.util.spec_from_file_location(
    "post_training_diagnostic", ROOT / "scripts/diagnostics/post_training_generation_diagnostic.py"
)
diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic)
comparison = diagnostic.comparison
resource_guard = diagnostic.resource_guard


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def guarded(operation):
    """Return comparison metrics, or the error an unrepaired API raises."""
    try:
        return operation()
    except (ValueError, RuntimeError) as error:
        return {"error": f"{type(error).__name__}: {error}"}


def load_model(path: Path, expected_sha256: str, spec: dict):
    import torch
    from llm_architecture.checkpointing import file_sha256
    from llm_architecture.model import LanguageModel, ModelConfig, count_parameters

    before = file_sha256(path)
    if before != expected_sha256:
        raise RuntimeError(f"Checkpoint identity mismatch for {path}: {before}")
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    config = payload["config"]["model"]
    assert config == spec["model"]
    model = LanguageModel(ModelConfig(**config))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    del payload
    gc.collect()
    assert model.lm_head.weight is model.token_embeddings.weight
    assert count_parameters(model) == spec["expected_parameters"]
    model.requires_grad_(False)
    model.eval()
    return model, before


def masked_cache_measurements(model, base: list[int], device: str) -> dict:
    """FP32 (no autocast), same inputs as the original isolated check."""
    import torch

    results: dict = {}
    with torch.inference_mode():
        ids = torch.tensor([base[:8]], device=device)
        continuation = torch.tensor([[base[8]]], device=device)
        pre = model(ids, use_cache=True)[2]
        full = model(torch.cat((ids, continuation), 1))[0][:, -1:]
        current_mask = torch.ones_like(continuation, dtype=torch.bool)
        results["default_current_mask"] = comparison(
            full, model(continuation, attention_mask=current_mask, past_key_values=pre, use_cache=True)[0]
        )
        results["explicit_offset"] = comparison(
            full,
            model(continuation, attention_mask=current_mask,
                  position_ids=torch.tensor([[8]], device=device), past_key_values=pre, use_cache=True)[0],
        )
        results["full_mask"] = guarded(lambda: comparison(
            full,
            model(continuation, attention_mask=torch.ones((1, 9), dtype=torch.bool, device=device),
                  past_key_values=pre, use_cache=True)[0],
        ))
        unmasked_cached = model(continuation, past_key_values=pre, use_cache=True)[0]
        results["unmasked_cached"] = comparison(full, unmasked_cached)

        # Left-padded batch: row 0 has two pads then 6 real tokens; row 1 has 8.
        prompt = torch.tensor([[0, 0, *base[:6]], base[:8]], device=device)
        prompt_mask = torch.tensor([[0, 0] + [1] * 6, [1] * 8], dtype=torch.bool, device=device)
        step = torch.tensor([[base[6]], [base[8]]], device=device)
        references = [model(torch.tensor([base[:7]], device=device))[0][0, -1],
                      model(torch.tensor([base[:9]], device=device))[0][0, -1]]
        batch_pre = model(prompt, attention_mask=prompt_mask, use_cache=True)[2]

        def padded_rows(mask):
            logits = model(step, attention_mask=mask, past_key_values=batch_pre, use_cache=True)[0][:, -1]
            return {f"row{index}": comparison(references[index][None], logits[index][None]) for index in range(2)}

        results["left_padded_legacy_current_mask"] = guarded(
            lambda: padded_rows(torch.ones((2, 1), dtype=torch.bool, device=device))
        )
        results["left_padded_full_mask"] = guarded(
            lambda: padded_rows(torch.cat((prompt_mask, torch.ones((2, 1), dtype=torch.bool, device=device)), 1))
        )
    return results


def unmasked_digests(model, base: list[int], device: str, autocast) -> dict:
    import torch

    digests = {}
    for precision, context in (("fp32", nullcontext), ("bf16_autocast", autocast)):
        with torch.inference_mode(), context():
            ids = torch.tensor([base[:24]], device=device)
            full = model(ids)[0]
            cache = model(ids[:, :16], use_cache=True)[2]
            cached = model(ids[:, 16:20], past_key_values=cache, use_cache=True)
            step = model(ids[:, 20:21], past_key_values=cached[2], use_cache=True)[0]
            digests[precision] = dict(full=tensor_digest(full), cached_chunk=tensor_digest(cached[0]),
                                      cached_step=tensor_digest(step))
    return digests


def generation_benchmark(model, cases: list[dict], device: str, autocast) -> list[dict]:
    import torch
    from llm_inference.generate import GenerationConfig, generate_sequences

    config = GenerationConfig(max_new_tokens=128, seed=42)
    rows = []
    with torch.inference_mode(), autocast():
        for case in cases:
            resource_guard()
            result = generate_sequences(model, torch.tensor([case["input_ids"]], device=device),
                                        eos_token_id=2, pad_token_id=0, stop_sequences=[[7]],
                                        config=config).sequences[0]
            rows.append(dict(id=case["id"], raw_token_ids=list(result.token_ids),
                             finish_reason=result.finish_reason, termination=result.termination_cause,
                             historical_token_match=list(result.token_ids) == case["raw_token_ids"]))
    return rows


def run_phase(directory: Path, phase: str) -> None:
    import torch
    import llm_architecture.model as model_module
    from llm_architecture.checkpointing import file_sha256
    from llm_tokenizer.tokenizer import load_tokenizer

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = read_json(RUN / "config/job.json")["spec"]
    tokenizer = load_tokenizer(spec["tokenizer"]["path"])
    base = tokenizer.encode("A computer stores information and follows instructions. ", add_special_tokens=False).ids
    base = (base * (32 // len(base) + 1))[:32]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.5)

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else nullcontext()

    output = directory / phase
    results = dict(phase=phase, historical_context=HISTORICAL_CONTEXT,
                   model_py=str(Path(model_module.__file__).resolve()),
                   model_py_sha256=file_sha256(model_module.__file__), torch_version=torch.__version__,
                   device=device, checkpoints={})
    checkpoints = (("final", RUN / "checkpoints/slot-0/checkpoint.pt", FINAL_SHA256),
                   ("baseline", Path(spec["initialization"]["path"]), spec["initialization"]["sha256"]))
    for label, path, expected in checkpoints:
        started = time.monotonic()
        model, digest = load_model(path, expected, spec)
        resource_guard()
        model.to(device)
        entry = dict(path=str(path), sha256_before=digest)
        entry["masked_cache"] = masked_cache_measurements(model, base, device)
        entry["unmasked_digests"] = unmasked_digests(model, base, device, autocast)
        cases = [json.loads(line) for line in
                 (DIAGNOSTIC / f"evidence/generation/historical-{label}.jsonl").read_text().splitlines()]
        generations = generation_benchmark(model, cases, device, autocast)
        output.mkdir(parents=True, exist_ok=True)
        (output / f"generations-{label}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in generations))
        entry["historical_token_matches"] = sum(row["historical_token_match"] for row in generations)
        entry["cases"] = len(generations)
        if device == "cuda":
            entry["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        entry["sha256_after"] = file_sha256(path)
        if entry["sha256_after"] != expected:
            raise RuntimeError(f"Checkpoint changed during validation: {path}")
        entry["seconds"] = time.monotonic() - started
        results["checkpoints"][label] = entry
        print(f"{phase}/{label}: {entry['historical_token_matches']}/{entry['cases']} historical matches; "
              f"default masked max_abs={entry['masked_cache']['default_current_mask']['max_absolute']:.6g}", flush=True)
    write_json(output / "results.json", results)


def compare(directory: Path) -> None:
    before = read_json(directory / "before/results.json")
    after = read_json(directory / "after/results.json")
    summary = dict(before_model_py_sha256=before["model_py_sha256"], after_model_py_sha256=after["model_py_sha256"],
                   checkpoints={})
    lines = ["# Masked KV-cache repair: before/after validation", "",
             f"- Before `model.py` SHA-256: `{before['model_py_sha256']}`",
             f"- After `model.py` SHA-256: `{after['model_py_sha256']}`", ""]
    for label in ("final", "baseline"):
        old, new = before["checkpoints"][label], after["checkpoints"][label]
        old_rows = [json.loads(s) for s in (directory / f"before/generations-{label}.jsonl").read_text().splitlines()]
        new_rows = [json.loads(s) for s in (directory / f"after/generations-{label}.jsonl").read_text().splitlines()]
        identical = sum(a["raw_token_ids"] == b["raw_token_ids"] and a["finish_reason"] == b["finish_reason"]
                        for a, b in zip(old_rows, new_rows, strict=True))
        entry = dict(
            masked_cache={key: dict(before=old["masked_cache"][key], after=new["masked_cache"][key])
                          for key in new["masked_cache"]},
            unmasked_digests_identical=old["unmasked_digests"] == new["unmasked_digests"],
            historical_matches=dict(before=old["historical_token_matches"], after=new["historical_token_matches"],
                                    cases=new["cases"]),
            before_after_identical_generations=identical,
            checkpoint_sha256_unchanged=old["sha256_before"] == old["sha256_after"] == new["sha256_before"]
            == new["sha256_after"],
        )
        summary["checkpoints"][label] = entry
        lines += [f"## {label}", "", "| Measurement | Before max abs | After max abs |", "|---|---:|---:|"]
        for key, value in entry["masked_cache"].items():
            def cell(metrics):
                if "error" in metrics:
                    return "raises: " + metrics["error"].split(":")[0]
                if "max_absolute" in metrics:
                    return f"{metrics['max_absolute']:.6g}"
                return " / ".join(f"{row}={metrics[row]['max_absolute']:.6g}" for row in sorted(metrics))
            lines.append(f"| {key} | {cell(value['before'])} | {cell(value['after'])} |")
        lines += ["", f"- Historical greedy token matches: before {old['historical_token_matches']}/{old['cases']}, "
                  f"after {new['historical_token_matches']}/{new['cases']}",
                  f"- Before/after identical generations: {identical}/{len(new_rows)}",
                  f"- Unmasked logits digests identical: {entry['unmasked_digests_identical']}",
                  f"- Checkpoint SHA-256 unchanged: {entry['checkpoint_sha256_unchanged']}", ""]
    lines += [
        "## Notes", "",
        "- All masked-cache measurements are FP32 without autocast on the default SDPA backend. "
        "Residual ~1e-5 differences equal the unmasked cached path's (different matmul shapes, same math).",
        "- `left_padded_legacy_current_mask` row0 passes a current-token mask over a cache that holds two "
        "padding tokens. By contract that form defines the cached prefix as fully valid, so the padding is "
        "attended to; the full `[batch, past + sequence]` mask is required for padded caches "
        "(`left_padded_full_mask`). The cache tuple stores no mask.",
        "- Generations use the original runtime (CUDA, BF16 autocast, greedy, 128 tokens, EOS 2, stop [[7]]); "
        "the generator compacts prompts and passes no model mask, so it never used the repaired path.",
        "",
    ]
    write_json(directory / "comparison.json", summary)
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("before", "after", "compare"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--historical-context", action="store_true",
                        help="use the installed .venv wheels instead of source components")
    args = parser.parse_args()
    directory = args.out.resolve()
    if not directory.is_relative_to(ALLOWED.resolve()) or directory == ALLOWED.resolve():
        parser.error(f"Use a dedicated child of {ALLOWED}")
    if args.phase == "compare":
        compare(directory)
    else:
        if (directory / args.phase / "results.json").exists():
            parser.error(f"{args.phase} results already exist in {directory}")
        run_phase(directory, args.phase)


if __name__ == "__main__":
    main()
