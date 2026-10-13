"""Indexed training-data selection for foundation jobs: exposure accounting, held-out validation
windows, near-duplicate families and the frozen selection record. CPU-only; no launch API."""

from bisect import bisect_left
from collections.abc import Iterable, Iterator
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import unicodedata

import numpy as np
import torch
from llm_architecture.checkpointing import file_sha256
from llm_data.token_views import overlapping_window_bounds

from codexa_workspace.hashing import fingerprint


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


