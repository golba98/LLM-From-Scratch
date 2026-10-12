"""CPU-only analysis and indexed-data proposals; deliberately no launch API."""

from bisect import bisect_left
from collections.abc import Iterable, Iterator
from collections import Counter, defaultdict
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import heapq
import json
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata

import numpy as np
import torch
import yaml

from llm_architecture.checkpointing import file_sha256
from llm_data.token_data import MemmapTokenDataset
from llm_data.token_views import IndexedTokenWindowDataset, overlapping_window_bounds
from llm_tokenizer.tokenizer import load_tokenizer
from llm_training.config import load_config

from .assets import resolve_input, output_root
from .chat_evaluation import fingerprint
from .foundation_diagnosis import code_identity


FORMAT = "codexa-foundational-pretraining-proposal-v1"
SOURCES = ("fineweb_edu", "wikipedia")
ARTIFACT_LIMIT = 64 * 1024**2
BASE_SHA256 = "808afec9923e3f00d91bcbef47643265ae5dbe6c5aeb4bd112c4e97be3d0a3a7"
TOKENIZER_SHA256 = "a710b68c646d4b375331f65ee164aed4456813cc1c1b7dcc7fa2bafc9450854e"


def write_json(path: Path, value: object) -> None:
    """Create a small owned artifact, refusing replacement."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def identity(path: Path) -> dict:
    """Record immutable-input metadata, without redundant huge-file hashing."""
    stat = path.stat()
    return dict(device=stat.st_dev, inode=stat.st_ino, bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def reconstruct_exposure(record: dict) -> np.ndarray:
    """Reproduce the original loader's base-seed draw and epoch-zero permutation."""
    if record["epoch"] != 0 or record["cursor"] > record["total_windows"]:
        raise ValueError("Unsupported historical cursor reconstruction.")
    if record["pytorch"] != torch.__version__:
        raise ValueError("Historical reconstruction requires the recorded Torch version.")
    generator = torch.Generator().manual_seed(record["seed"])
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    prefix = torch.randperm(record["total_windows"], generator=generator)[:record["cursor"]].numpy()
    digest = hashlib.sha256(prefix.astype("<i8").tobytes()).hexdigest()
    if digest != record["prefix_sha256"]:
        raise ValueError("Historical sampled-position fingerprint mismatch.")
    return np.sort(prefix.astype("<u4"))


def index_rows(path: Path, *, token_count: int, expected_sha256: str,
               observations: dict) -> Iterator[dict]:
    """Stream and hash the existing line-oriented JSON index, validating continuity."""
    digest = hashlib.sha256()
    previous_end = 0
    count = 0
    source_tokens = Counter()
    source_documents = Counter()
    with path.open("rb") as stream:
        first = stream.readline()
        digest.update(first)
        if first.strip() != b"[":
            raise ValueError("Index must begin with a JSON array.")
        closed = False
        for raw in stream:
            digest.update(raw)
            line = raw.strip()
            if line == b"]":
                closed = True
                tail = stream.read()
                digest.update(tail)
                if tail.strip():
                    raise ValueError("Unexpected trailing index content.")
                break
            if line.endswith(b","):
                line = line[:-1]
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Document index row must be an object.")
            start, end = row.get("token_start"), row.get("token_end")
            if (type(start) is not int or type(end) is not int or start != previous_end
                    or end <= start or end > token_count
                    or row.get("document_ordinal") != count
                    or row.get("total_stored_token_count") != end - start
                    or row.get("source") not in SOURCES
                    or not isinstance(row.get("document_id"), str)):
                raise ValueError(f"Invalid document index geometry at ordinal {count}.")
            source_tokens[row["source"]] += end - start
            source_documents[row["source"]] += 1
            previous_end = end
            count += 1
            yield row
        if not closed or previous_end != token_count:
            raise ValueError("Index is incomplete or token coverage differs.")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("Document index checksum mismatch.")
    observations.update(documents=count, tokens=previous_end,
                        tokens_by_source=dict(source_tokens), documents_by_source=dict(source_documents),
                        sha256=digest.hexdigest())


def interval_exposed(start: int, end: int, context: int,
                     window_count: int, exposed: list[int]) -> bool:
    """Include conditioning tokens as well as shifted targets in exposure checks."""
    first, stop = overlapping_window_bounds(start, end, context, window_count)
    cursor = bisect_left(exposed, first)
    return cursor < len(exposed) and exposed[cursor] < stop


def select_candidates(rows: Iterable[dict], *, context: int, window_count: int,
                      exposed: list[int], pool_per_source: int, seed: int) -> list[dict]:
    """Select bounded lowest-hash candidates from all eligible index documents."""
    heaps = {source: [] for source in SOURCES}
    for row in rows:
        start, end = row["token_start"], row["token_end"]
        aligned = ((start + context - 1) // context) * context
        if (end - start > 16384 or aligned + context + 1 > end
                or interval_exposed(start, end, context, window_count, exposed)):
            continue
        key = f'{seed}:{row["source"]}:{row["document_id"]}:{row["document_ordinal"]}'
        rank = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        heap = heaps[row["source"]]
        entry = (-rank, row["document_ordinal"], dict(row, selection_rank=rank,
                                                     window_id=aligned // context))
        if len(heap) < pool_per_source:
            heapq.heappush(heap, entry)
        elif entry[:2] > heap[0][:2]:
            heapq.heapreplace(heap, entry)
    return sorted([entry[2] for heap in heaps.values() for entry in heap],
                  key=lambda row: (row["selection_rank"], row["document_ordinal"]))


def normalized_text(text: str) -> str:
    """Define the proposal's exact-content normalization, not semantic equivalence."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def content_features(ids: np.ndarray, tokenizer) -> tuple[dict, set[int]]:
    """Validate a bounded complete document and summarize duplicate evidence."""
    values = ids.tolist()
    if not values or max(values) >= 16384:
        raise ValueError("Invalid candidate token IDs.")
    text = tokenizer.decode(values, skip_special_tokens=False)
    if tokenizer.encode(text, add_special_tokens=False).ids != values:
        raise ValueError("Candidate tokenizer round-trip failed.")
    normalized = normalized_text(text)
    words = normalized.split()
    shingles = {int.from_bytes(hashlib.sha256(" ".join(words[i:i+5]).encode()).digest()[:8], "big")
                for i in range(max(0, len(words)-4))}
    features = dict(normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    token_bytes_sha256=hashlib.sha256(ids.astype("<u2").tobytes()).hexdigest(),
                    words=len(words), replacement_characters=text.count("\ufffd"),
                    roundtrip=True, excerpt=text[:240],
                    sketch=sorted(shingles)[:64])
    return features, shingles


def duplicate_groups(rows: list[dict], shingles: list[set[int]]) -> tuple[list[int], list[dict]]:
    """Group bounded candidates by exact text or high-overlap five-word shingles."""
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def join(i, j):
        parent[find(j)] = find(i)
    exact = {}
    buckets = defaultdict(list)
    edges = []
    for i, row in enumerate(rows):
        digest = row["normalized_sha256"]
        if digest in exact:
            j = exact[digest]
            join(i, j)
            edges.append(dict(left=i, right=j, kind="normalized_exact"))
        else:
            exact[digest] = i
        candidates = Counter(j for value in row["sketch"] for j in buckets[value])
        for j, shared in candidates.items():
            if shared < 16 or min(row["words"], rows[j]["words"]) / max(row["words"], rows[j]["words"]) < .8:
                continue
            union = shingles[i] | shingles[j]
            similarity = len(shingles[i] & shingles[j]) / len(union) if union else 0
            if similarity >= .8:
                join(i, j)
                edges.append(dict(left=i, right=j, kind="near_duplicate", jaccard=similarity))
        for value in row["sketch"]:
            buckets[value].append(i)
    return [find(i) for i in range(len(rows))], edges


def select_validation(rows: list[dict], groups: list[int], per_source: int) -> list[int]:
    """Keep one representative per known family; reject any exposed family."""
    blocked = {groups[i] for i, row in enumerate(rows) if row["historical_overlap"]}
    selected, used = [], set()
    counts = Counter()
    for i in sorted(range(len(rows)), key=lambda i: rows[i]["selection_rank"]):
        source = rows[i]["source"]
        if groups[i] in used or groups[i] in blocked or counts[source] >= per_source:
            continue
        selected.append(i)
        used.add(groups[i])
        counts[source] += 1
    if any(counts[source] != per_source for source in SOURCES):
        raise ValueError("Insufficient independent candidate families for the declared split.")
    return selected


def training_selection(window_count: int, context: int, exposed: np.ndarray,
                       excluded_intervals: list[tuple[int, int]]) -> np.ndarray:
    """Exclude historical positions and every selected-family conditioning/target overlap."""
    keep = np.ones(window_count, dtype=bool)
    keep[exposed] = False
    for start, end in excluded_intervals:
        first, stop = overlapping_window_bounds(start, end, context, window_count)
        keep[first:stop] = False
    return np.flatnonzero(keep).astype("<u4")


def analyze_outputs(directory: Path) -> dict:
    """Summarize immutable recorded behavior; never generate semantic judgments."""
    result = {}
    for name in ("base", "repair", "balanced", "original_2h", "END3"):
        path = directory/name/"outputs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != 240 or len({row["id"] for row in rows}) != 240:
            raise ValueError("Incomplete or duplicate baseline case coverage.")
        lengths = sorted(row["generated_tokens"] for row in rows)
        categories = defaultdict(list)
        for row in rows:
            categories[row["category"]].append(row)
        counts = {key: sum(bool(row["metrics"].get(key)) for row in rows)
                  for key in ("empty", "severe_degeneration", "ended_with_END", "ended_with_EOS",
                              "correct_END", "premature_END_candidate")}
        counts["forced_cutoff"] = sum(row["termination"] == "length" for row in rows)
        counts["role_failures"] = sum(row["metrics"]["unexpected_role_tokens"] > 0 for row in rows)
        strict = [row for row in rows if row["category"] != "retention"]
        result[name] = dict(input_sha256=file_sha256(path), cases=len(rows),
            generated_tokens=sum(lengths), length=dict(minimum=min(lengths), maximum=max(lengths),
            median=statistics.median(lengths), mean=statistics.mean(lengths), p90=lengths[215]),
            counts=counts, objective_cases=len(strict), objective_passes=sum(row["metrics"]["task_pass"] for row in strict),
            mean_repeated_token_4gram=statistics.mean(row["metrics"]["token_repeated_4gram_rate"] for row in rows),
            categories={category: dict(cases=len(items), strict_passes=None if category == "retention" else
                sum(row["metrics"]["task_pass"] for row in items)) for category, items in categories.items()},
            semantic_correctness=None, coherence=None, human_review="pending",
            automatic_signals_are_not_semantic_judgments=True)
    return result


def conservative_answer_check(case: dict, response: str) -> dict:
    """Recognize only an entire bounded answer, never a keyword in longer prose."""
    expected = str(case["expected"])
    reference_verified = case["category"] != "arithmetic"
    if case["category"] == "arithmetic":
        match = re.search(r"(-?\d+) (plus|minus|times|divided by) (-?\d+)", case["prompt"])
        if match:
            left, operation, right = match.groups()
            a, b = Fraction(left), Fraction(right)
            answer = {"plus":lambda:a+b, "minus":lambda:a-b,
                      "times":lambda:a*b, "divided by":lambda:a/b}[operation]()
            reference_verified = answer == Fraction(expected)
        if not reference_verified:
            return dict(whole_answer_recognized=False, reference_consistency=None,
                        reference_verified=False, human_semantic_judgment=None)
    # Full matching rejects any subsequent contradiction, explanation or repetition.
    pattern = r"(?:the (?:answer|result) is |it is |answer: )?([-+]?\d+(?:\.\d+)?|[A-Za-z]+(?: [A-Za-z]+)?)[.!]?"
    match = re.fullmatch(pattern, response.strip(), flags=re.IGNORECASE)
    consistency = None
    if match:
        answer = match.group(1)
        if case["category"] == "arithmetic":
            try:
                consistency = Decimal(answer) == Decimal(expected)
            except InvalidOperation:
                consistency = None
        else:
            consistency = normalized_text(answer) == normalized_text(expected)
    return dict(whole_answer_recognized=match is not None, reference_consistency=consistency,
                reference_verified=reference_verified if case["category"] == "arithmetic" else None,
                reference_basis="independent arithmetic calculation" if case["category"] == "arithmetic" else
                    "existing development reference; independent human review remains pending",
                human_semantic_judgment=None)


def objective_answer_analysis(root: Path, directory: Path) -> dict:
    """Add narrowly recognized answer evidence without interpreting unreviewed prose."""
    suite_path = root/"outputs/readiness/codexa-934m-20261010-v1/prepared-v3/development-suite.json"
    suite = json.loads(suite_path.read_text())
    baseline = root/"outputs/qualification/codexa-934m-20261011-v1/baseline-generations"
    comparison = json.loads((baseline/"comparison.json").read_text())
    if fingerprint(suite) != comparison["plan"]["suite_sha256"] or any(case.get("protected") for case in suite):
        raise ValueError("Development suite drift or acceptance exposure.")
    cases = {case["id"]:case for case in suite}
    results = {}
    for name in ("base", "repair", "balanced", "original_2h", "END3"):
        rows = []
        for line in (baseline/name/"outputs.jsonl").read_text().splitlines():
            output = json.loads(line)
            case = cases[output["id"]]
            if case["category"] in ("arithmetic", "factual"):
                rows.append(dict(case_id=case["id"], category=case["category"],
                                 **conservative_answer_check(case, output["response"])))
        results[name] = dict(rows=rows, consistent_whole_answers=sum(row["reference_consistency"] is True for row in rows),
            inconsistent_recognized_answers=sum(row["reference_consistency"] is False for row in rows),
            unrecognized_or_unverified=sum(row["reference_consistency"] is None for row in rows),
            semantic_scores=None)
    report = dict(development_suite_sha256=comparison["plan"]["suite_sha256"], results=results,
        metric="conservative whole-answer reference consistency; NOT a broad semantic accuracy score",
        unrecognized_prose_is_unknown_not_incorrect=True, outputs_regenerated=False)
    write_json(directory/"objective-answer-analysis.json", report)
    return report


def budgets() -> dict:
    """Calculate targets separately from wall time; all speed inputs are historical."""
    per_update = 2048 * 32
    original = 655360000
    remaining = 12819847234 - original
    speed = 8003
    return dict(targets_per_update=per_update, pilot_targets=256 * per_update,
                pilot_compute_seconds=256 * per_update / speed,
                historical_context_tokens_per_second=speed,
                context_and_targets_equal_for_unpadded_full_windows=True,
                ten_hour_scenarios=[dict(overhead_fraction=fraction,
                    targets=int(speed * 36000 * (1-fraction)),
                    updates=int(speed * 36000 * (1-fraction) // per_update),
                    total_tokens_per_parameter=(original + speed * 36000 * (1-fraction)) / 934356480,
                    remaining_position_fraction=speed * 36000 * (1-fraction) / remaining)
                    for fraction in (0., .1, .2)],
                legacy_peak_working_bytes=7 * 5636391032 + 1024**3,
                minimum_disk_free_bytes=180 * 1024**3,
                disk_policy="Existing preflight: max(180 GiB, 100 GiB + planned peak); no gate changed.")


def pilot_spec(model_config: dict, selection_sha256: str, config_path: Path,
               initializer: dict, tokenizer_path: Path) -> tuple[dict, dict]:
    """Return a schema-valid YAML and explicitly blocked new-phase specification."""
    config = dict(model=deepcopy(model_config), training=dict(micro_batch_size=1,
        gradient_accumulation_steps=32, learning_rate=2e-5, weight_decay=.1,
        warmup_steps=20, max_steps=256, gradient_clip=1., precision="bf16",
        checkpoint_interval=64, evaluation_interval=64, seed=42))
    spec = dict(format=FORMAT, status="PROPOSED_NOT_EXECUTABLE", execution_enabled=False,
        phase="new continued next-token pretraining", initialization=dict(initializer,
            policy="weights_only_fresh_optimizer_and_scheduler", provisional=True,
        exact_resume_of_original_base=False), architecture=model_config,
        model_config_sha256=fingerprint(model_config),
        tokenizer_path=str(tokenizer_path), tokenizer_sha256=TOKENIZER_SHA256,
        configuration_file=str(config_path), data_selection_sha256=selection_sha256,
        objective="all_valid_next_token_targets", optimizer=dict(name="adamw8bit", betas=[.9, .95], epsilon=1e-8),
        minimum_learning_rate=2e-6, gradient_checkpointing=True, fp32_master_weights=True,
        loss_normalization="supervised_tokens", target_token_weights={},
        wall_clock_seconds=3600, maximum_targets=256*2048*32, max_epochs=1,
        finalization_reserve_seconds=180, checkpoint_backend="legacy", checkpoint_slots=6,
        checkpoint_roles=["latest", "previous", "best_validation", "best_behavioral", "final"],
        budgets=budgets(), evaluation=dict(validation_windows_per_source=128,
            cadence_updates=64, report_source_losses=True,
            generation_cadence_updates=128, generation_cases=48,
            generation_max_new_tokens=128, generation_greedy=True, repetition_penalty=1,
            paired_analysis_unit="24 concepts; average neutral/direct within concept before comparison",
            bootstrap_samples=2000, bootstrap_seed=42,
            combined_loss_weights="declared corpus token shares, not equal sample shares",
            no_update_control=True, generation_protocol="frozen foundational neutral/direct cases; new pilot approval required",
            human_semantic_review_required=True, acceptance_suite_used=False),
        quality_guards=dict(nonfinite="stop immediately", failed_checkpoint="stop; retain last verified recovery",
            gradient_norm_threshold=10., gradient_norm_patience=3,
            source_validation_CE_increase=.1, validation_patience=2,
            severe_degeneration_increase_max=.05, meaningful_loss_signal_CE_decrease=.02,
            extended_training_gate=dict(reviewed_net_concept_gain_minimum=3,
                paired_95_percent_cluster_lower_bound_strictly_positive=True,
                reviewed_semantic_regression_forbidden=True,
                loss_alone_is_insufficient=True)),
        interruption=dict(safe_optimizer_boundaries=True, incomplete_accumulation="rollback counters/cursor and discard gradients",
            hard_kill="last verified full checkpoint only", budget="original deadline persists; never reset",
            reconstruction="new-phase sampler identity/epoch/cursor, optimizer, scheduler, RNG, counters required"),
        launch_blockers=["neutral-format evidence and genuine review", "indexed-view launch wiring",
            "full-shape legacy training headroom", "live resource gates", "precise pilot approval"],
        resource_requirements=dict(minimum_host_available_bytes=16*1024**3,
            minimum_disk_free_bytes=180*1024**3, GPU="RTX 4080; fresh measured full-shape margin required",
            VRAM_planning_estimate_MiB=13928,
            VRAM_estimate_basis="original base reserved peak 12392 MiB plus provisional 1536 MiB margin; not a new measurement",
            visible_kitty_viewer="100x22 required at future launch"),
        no_launch_API=True)
    return config, spec


def validate_proposal(spec: dict) -> None:
    """Reject accidental promotion, unsafe initialization or incompatible budgets."""
    if (spec.get("format") != FORMAT or spec.get("execution_enabled") is not False
            or spec.get("status") != "PROPOSED_NOT_EXECUTABLE"
            or spec["initialization"]["policy"] != "weights_only_fresh_optimizer_and_scheduler"
            or spec["initialization"]["sha256"] != BASE_SHA256
            or spec["tokenizer_sha256"] != TOKENIZER_SHA256
            or spec["maximum_targets"] != 256*2048*32
            or spec["checkpoint_backend"] != "legacy"
            or spec["resource_requirements"]["minimum_disk_free_bytes"] < 180*1024**3
            or spec["model_config_sha256"] != fingerprint(spec["architecture"])):
        raise ValueError("Unsafe or drifted offline proposal.")


def validate_selection(directory: Path) -> dict:
    """Verify checksummed view arrays and complete positional exclusion before use."""
    record = json.loads((directory/"selection-manifest.json").read_text())
    if (record.get("format") != "codexa-indexed-pretraining-selection-v1"
            or record.get("execution_enabled") is not False
            or fingerprint({k:v for k, v in record.items() if k != "selection_sha256"}) != record["selection_sha256"]):
        raise ValueError("Selection manifest identity drift.")
    arrays = {}
    for name, artifact in record["view_arrays"].items():
        path = directory/name
        if path.resolve() != Path(artifact["path"]).resolve() or file_sha256(path) != artifact["sha256"]:
            raise ValueError("Selection array identity drift.")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 1 or values.dtype != np.dtype("<u4") or np.any(values[1:] <= values[:-1]):
            raise ValueError("Selection arrays must be sorted unique uint32 vectors.")
        if len(values) and int(values[-1]) >= record["index_observations"]["windows"]:
            raise ValueError("Selection array escapes the corpus window population.")
        arrays[name] = values
    train = arrays["train_window_ids.npy"]
    validation = arrays["validation_window_ids.npy"]
    historical = arrays["historical_window_ids.npy"]
    if (len(train) != record["eligible_training_windows"] or len(validation) != record["held_out_windows"]
            or len(historical) != record["historical_windows"]
            or np.intersect1d(train, validation).size or np.intersect1d(train, historical).size):
        raise ValueError("Training/evaluation or historical position leakage.")
    for start, end in record["excluded_intervals"]:
        first, stop = overlapping_window_bounds(start, end, record["context_length"],
                                               record["index_observations"]["windows"])
        position = np.searchsorted(train, first)
        if position < len(train) and train[position] < stop:
            raise ValueError("Held-out document context leaked into training.")
    return record


def targets_before(position: int, keep: np.ndarray, prefix: np.ndarray, context: int) -> int:
    """Count selected shifted-target positions strictly before a corpus offset."""
    if position <= 0:
        return 0
    window, partial = divmod(position-1, context)
    full = int(prefix[min(window, len(keep))]) * context
    return full + (partial if window < len(keep) and keep[window] else 0)


def prepare_pilot_exposure(directory: Path) -> dict:
    """Count actual source exposure using index metadata, never reading token payloads."""
    selection = validate_selection(directory)
    manifest = json.loads(Path(selection["canonical_manifest_path"]).read_text())
    if file_sha256(Path(selection["canonical_manifest_path"])) != selection["canonical_manifest_sha256"]:
        raise ValueError("Canonical manifest drift.")
    train = np.load(directory/"train_window_ids.npy", mmap_mode="r", allow_pickle=False)
    generator = torch.Generator().manual_seed(selection["seed"])
    torch.empty((), dtype=torch.int64).random_(generator=generator)
    order = torch.randperm(len(train), generator=generator)[:256*32].numpy()
    pilot = np.asarray(train[order], dtype="<u4")
    np.save(directory/"pilot_window_order.npy", pilot, allow_pickle=False)
    window_count = selection["index_observations"]["windows"]
    train_mask = np.zeros(window_count, dtype=bool)
    pilot_mask = np.zeros(window_count, dtype=bool)
    train_mask[train] = True
    pilot_mask[pilot] = True
    train_prefix = np.concatenate([np.zeros(1, dtype="uint32"), np.cumsum(train_mask, dtype="uint32")])
    pilot_prefix = np.concatenate([np.zeros(1, dtype="uint32"), np.cumsum(pilot_mask, dtype="uint32")])
    totals, pilot_totals = Counter(), Counter()
    index = Path(selection["canonical_manifest_path"]).parent/"train_index.json"
    observations = {}
    before = identity(index)
    print("One metadata-only accounting pass: exact eligible and proposed-pilot source targets", flush=True)
    for row in index_rows(index, token_count=manifest["train_token_count"],
            expected_sha256=selection["index_sha256"], observations=observations):
        start, end = row["token_start"], row["token_end"]
        totals[row["source"]] += (targets_before(end, train_mask, train_prefix, 2048)
                                 - targets_before(start, train_mask, train_prefix, 2048))
        pilot_totals[row["source"]] += (targets_before(end, pilot_mask, pilot_prefix, 2048)
                                       - targets_before(start, pilot_mask, pilot_prefix, 2048))
    if identity(index) != before or sum(totals.values()) != selection["eligible_target_positions"]:
        raise ValueError("Source accounting drift or incomplete target coverage.")
    if sum(pilot_totals.values()) != 256*32*2048:
        raise ValueError("Proposed pilot target count differs from fixed update budget.")
    record = dict(selection_sha256=selection["selection_sha256"], seed=selection["seed"],
        pilot_window_order_sha256=file_sha256(directory/"pilot_window_order.npy"),
        eligible_targets_by_source=dict(totals), pilot_targets_by_source=dict(pilot_totals),
        eligible_source_token_shares={k:v/sum(totals.values()) for k,v in totals.items()},
        pilot_source_token_shares={k:v/sum(pilot_totals.values()) for k,v in pilot_totals.items()},
        unique_pilot_windows=len(np.unique(pilot)), pilot_windows=len(pilot),
        semantic_uniqueness=None, sampler="Existing DataLoader base-seed draw then RandomSampler without replacement",
        tokenizer_or_payload_read=False, index_sha256=observations["sha256"],
        execution_authorized=False)
    write_json(directory/"pilot-source-exposure.json", record)
    return record


def prepare_review_queue(root: Path, directory: Path) -> dict:
    """Prioritize existing anonymous cases without mapping aliases or deciding answers."""
    page = root/"outputs/diagnosis/codexa-934m-foundation-20261011-v1/review/fresh-blinded-baseline"
    package = json.loads((page/"package.json").read_text())
    selected = [item["id"] for item in package["items"] if item["id"].split("::")[-1] in
                {"dev-v2-048", "dev-v2-049", "dev-v2-050", "dev-v2-051",
                 "dev-v2-152", "dev-v2-153", "dev-v2-200", "dev-v2-201"}]
    instructions = [item["id"] for item in package["items"] if item["group"] == "instruction"]
    record = dict(package_path=str(page/"package.json"), package_file_sha256=file_sha256(page/"package.json"),
        review_page=str(page/"review.html"), first_matched_semantic_cases=selected,
        instruction_cases=instructions, decisions_complete=False,
        instructions="Open the existing page; filter factual, multi_turn, retention, then instruction. Judge the whole answer, reject contradictions, leave uncertainty pending. Use a real reviewer name and export JSON before closing; there is no autosave. Complete remaining cases before admission/model selection.",
        private_mapping_read=False, new_responses_generated=0, protected_acceptance_used=False)
    write_json(directory/"review-queue.json", record)
    (directory/"REVIEW_QUEUE.md").write_text(
        "# Existing blinded review queue\n\n[Open the unchanged anonymous review](" + str(page/"review.html") + ")\n\n"
        + record["instructions"] + "\n\nFirst review the 40 factual/context/retention responses listed below, then the 60 instruction responses. "
        "Automated exact-format failure is separate from semantic correctness. Neither a keyword nor a raw END is an approval. "
        "Do not open private-mapping.json.\n\n" + "\n".join("- " + value for value in selected)
        + "\n\nThe machine-readable queue identifies every instruction item. No responses, aliases, approvals or reviewer fields were regenerated.\n")
    return record


def build(root: Path, directory: Path, *, pool_per_source: int = 512) -> dict:
    """Prepare small offline evidence; no checkpoint loading, GPU use or launch."""
    started = time.monotonic()
    if torch.cuda.is_initialized():
        raise RuntimeError("Preparation must not initialize CUDA.")
    if Path(sys.prefix).resolve() != (root/".venv").resolve():
        raise RuntimeError("Use the existing generative environment.")
    if not directory.resolve().is_relative_to(output_root(root)):
        raise ValueError("Preparation must stay within configured workspace outputs.")
    directory.mkdir(parents=True, exist_ok=False)
    previous = root/"outputs/diagnosis/codexa-934m-foundation-20261011-v1"
    job = previous/"job-v2"
    frozen = json.loads((job/"plan.json").read_text())
    if code_identity(root) != frozen["code_identity"]:
        raise ValueError("Frozen diagnostic implementation already drifted.")
    protected_job = {str(job/name): file_sha256(job/name) for name in ("plan.json", "suite.json", "approval.json")}
    baseline = root/"outputs/qualification/codexa-934m-20261011-v1/baseline-generations"
    write_json(directory/"behavior-analysis.json", analyze_outputs(baseline))
    history = json.loads((previous/"training-history.json").read_text())
    for path, digest in history["source_hashes"].items():
        if file_sha256(Path(path)) != digest:
            raise ValueError("Historical training evidence identity changed.")
    write_json(directory/"history-evidence.json", dict(source_sha256=file_sha256(previous/"training-history.json"),
        record=history, evidence_hierarchy=dict(token_count="logged and checkpoint metadata",
        unique_windows="same-version conditional reconstruction", undertraining="plausible hypothesis",
        neutral_language_ability="unknown; approved diagnostic not executed")))
    reconstruction = json.loads((previous/"sampler-reconstruction.json").read_text())
    exposure = reconstruct_exposure(reconstruction)
    np.save(directory/"historical_window_ids.npy", exposure, allow_pickle=False)
    corpus = resolve_input(root, "data/tokenized/base-v1/mixed")
    manifest_path = corpus/"token_data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest["tokenizer_sha256"] != TOKENIZER_SHA256 or manifest["dtype"] != "uint16"
            or manifest["context_length"] != 2048 or manifest["model_vocab_size"] != 16384):
        raise ValueError("Incompatible corpus manifest.")
    tokenizer_path = resolve_input(root, "checkpoints/tokenizer-base-v1/tokenizer.json")
    if file_sha256(tokenizer_path) != TOKENIZER_SHA256:
        raise ValueError("Tokenizer identity differs.")
    tokenizer = load_tokenizer(tokenizer_path)
    token_file = corpus/"train.bin"
    index = corpus/"train_index.json"
    before = {str(path): identity(path) for path in (token_file, index, manifest_path, tokenizer_path)}
    if token_file.stat().st_size != manifest["train_token_count"]*2:
        raise ValueError("Corpus file size disagrees with declared uint16 token count.")
    context, window_count = 2048, manifest["complete_sequences"]
    if window_count != reconstruction["total_windows"]:
        raise ValueError("Historical exposure belongs to a different window population.")
    exposed = exposure.tolist()
    first_pass = {}
    print("Index pass 1: deterministic bounded validation candidates", flush=True)
    candidates = select_candidates(index_rows(index, token_count=manifest["train_token_count"],
        expected_sha256=manifest["output_checksums"]["index"], observations=first_pass),
        context=context, window_count=window_count, exposed=exposed,
        pool_per_source=pool_per_source, seed=42)
    if (first_pass["documents"] != manifest["train_document_count"]
            or first_pass["tokens_by_source"] != manifest["tokens_by_source"]
            or first_pass["documents_by_source"] != manifest["documents_by_source"]):
        raise ValueError("Index source counts disagree with manifest.")
    candidate_keys = {(row["source"], row["document_id"]) for row in candidates}
    occurrences = defaultdict(list)
    second_pass = {}
    print(f"Index pass 2: all occurrences of {len(candidate_keys)} candidate source identities", flush=True)
    for row in index_rows(index, token_count=manifest["train_token_count"],
            expected_sha256=manifest["output_checksums"]["index"], observations=second_pass):
        key = row["source"], row["document_id"]
        if key in candidate_keys:
            occurrences[key].append((row["token_start"], row["token_end"]))
            if len(occurrences[key]) > 10000:
                raise ValueError("Unbounded candidate source identity occurrences; refuse split.")
    if second_pass != first_pass:
        raise ValueError("Index changed between metadata passes.")
    first_pass["windows"] = window_count
    tokens = np.memmap(token_file, mode="r", dtype="<u2")
    shingle_sets = []
    print("Bounded candidate content checks; original token stream remains read-only", flush=True)
    payload_read_tokens = 0
    for row in candidates:
        intervals = occurrences[(row["source"], row["document_id"])]
        row["occurrences"] = intervals
        row["historical_overlap"] = any(interval_exposed(a, b, context, window_count, exposed) for a, b in intervals)
        ids = np.asarray(tokens[row["token_start"]:row["token_end"]])
        features, shingles = content_features(ids, tokenizer)
        payload_read_tokens += len(ids)
        row.update(features)
        shingle_sets.append(shingles)
    groups, edges = duplicate_groups(candidates, shingle_sets)
    del shingle_sets
    selected = select_validation(candidates, groups, 128)
    selected_groups = {groups[i] for i in selected}
    intervals = sorted({tuple(interval) for i, row in enumerate(candidates)
                        if groups[i] in selected_groups for interval in row["occurrences"]})
    train_ids = training_selection(window_count, context, exposure, intervals)
    validation_ids = np.array(sorted(candidates[i]["window_id"] for i in selected), dtype="<u4")
    if np.intersect1d(train_ids, validation_ids).size:
        raise ValueError("Validation leaked into training view.")
    np.save(directory/"train_window_ids.npy", train_ids, allow_pickle=False)
    np.save(directory/"validation_window_ids.npy", validation_ids, allow_pickle=False)
    selected_docs = [dict(candidates[i], duplicate_group=groups[i]) for i in selected]
    write_json(directory/"validation-documents.json", selected_docs)
    write_json(directory/"duplicate-checks.json", dict(candidate_documents=len(candidates), edges=edges,
        selected_document_ids=[row["document_id"] for row in selected_docs],
        algorithm="NFKC/casefold/whitespace normalized exact hashes; bottom-64 five-word shingle candidates >=16 shared, full Jaccard >=0.8 and word-length ratio >=0.8",
        limitations=["Bounded candidate pool only; full-corpus exact/near duplicates not excluded.",
                    "High boilerplate overlap can group unrelated documents; this is conservative exclusion."] ))
    dataset = MemmapTokenDataset(token_file, dtype="uint16", context_length=context)
    validation = IndexedTokenWindowDataset(dataset, validation_ids, vocab_size=16384)
    for i in range(len(validation)):
        inputs, labels = validation[i]
        if not torch.equal(inputs[1:], labels[:-1]):
            raise ValueError("Selected validation window has incorrect target shift.")
    for key, expected in before.items():
        if identity(Path(key)) != expected:
            raise ValueError("Canonical input changed during preparation.")
    for key, digest in protected_job.items():
        if file_sha256(Path(key)) != digest:
            raise ValueError("Approved diagnostic artifact changed during preparation.")
    if code_identity(root) != frozen["code_identity"] or torch.cuda.is_initialized():
        raise ValueError("Diagnostic implementation changed or CUDA initialized.")
    arrays = {name: dict(path=str(directory/name), sha256=file_sha256(directory/name))
              for name in ("train_window_ids.npy", "validation_window_ids.npy", "historical_window_ids.npy")}
    selection = dict(format="codexa-indexed-pretraining-selection-v1", execution_enabled=False,
        canonical_manifest_path=str(manifest_path), canonical_manifest_sha256=file_sha256(manifest_path),
        corpus_path=str(token_file), corpus_sha256=manifest["output_checksums"]["train"],
        corpus_full_payload_hash_refreshed=False, corpus_input_identity=before[str(token_file)],
        index_sha256=first_pass["sha256"], tokenizer_sha256=TOKENIZER_SHA256,
        context_length=context, seed=42, sampling="new seeded no-replacement traversal of sorted eligible window view",
        view_arrays=arrays, historical_windows=len(exposure), eligible_training_windows=len(train_ids),
        eligible_target_positions=len(train_ids)*context, held_out_windows=len(validation_ids),
        held_out_targets=len(validation_ids)*context, held_out_windows_by_source={source:128 for source in SOURCES},
        excluded_document_occurrences=len(intervals), excluded_intervals=intervals,
        historical_reconstruction_sha256=file_sha256(previous/"sampler-reconstruction.json"),
        historical_exposure_limitation=reconstruction["limitations"],
        validation_exposure_class="isolated under reconstructed base-lineage positional exposure; NOT guaranteed historically/content-unseen",
        population="source-balanced 128 long-enough documents/source; full aligned windows within documents; document length <=16384 tokens",
        source_loss_weights=manifest["source_percentages"], index_observations=first_pass,
        bounded_payload_tokens_read=payload_read_tokens, canonical_corpus_modified=False,
        semantic_unique_tokens=None, near_duplicate_independence=False,
        licensing="Use existing provenance/license records; no new licensing determination.")
    selection["selection_sha256"] = fingerprint(selection)
    write_json(directory/"selection-manifest.json", selection)
    validate_selection(directory)
    # The original audited configuration is authoritative, not conversational loss rankings.
    model_config = history["selected_run"]["model_config"]
    initializer = json.loads((baseline/"comparison.json").read_text())["plan"]["candidates"]["base"]
    if initializer["sha256"] != BASE_SHA256:
        raise ValueError("Unexpected base initializer identity.")
    config_path = directory/"foundation-pilot.proposed.yaml"
    config, spec = pilot_spec(model_config, selection["selection_sha256"], config_path, initializer, tokenizer_path)
    with config_path.open("x") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)
    load_config(config_path)
    spec["configuration_sha256"] = file_sha256(config_path)
    validate_proposal(spec)
    write_json(directory/"foundation-pilot.proposed.json", spec)
    write_json(directory/"budget-estimates.json", budgets())
    prepare_pilot_exposure(directory)
    prepare_review_queue(root, directory)
    objective_answer_analysis(root, directory)
    write_json(directory/"preservation.json", dict(canonical_input_identities_before=before,
        unchanged=True, approved_job_hashes_unchanged=protected_job, diagnostic_code_unchanged=True,
        cuda_initialized=False, checkpoint_tensors_loaded=False))
    total = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
    if total > ARTIFACT_LIMIT:
        raise RuntimeError("Preparation exceeded its persistent artifact allowance; no cleanup performed.")
    status = dict(state="CPU_PREPARATION_COMPLETE_EXECUTION_BLOCKED", artifact_bytes=total,
        elapsed_seconds=time.monotonic()-started, CUDA_initialized=False,
        recommended_training_strategy="EXISTING EVIDENCE IS INSUFFICIENT TO CHOOSE — NEUTRAL-FORMAT EVALUATION REQUIRED")
    write_json(directory/"completion.json", status)
    return status
