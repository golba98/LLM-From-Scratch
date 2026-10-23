"""Order-independent document identity, deduplication, and split governance."""

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
import hashlib
import math
import re

from llm_data.data.cleaning import clean_text, text_sha256
from llm_data.data.io import TextDocument


GOVERNANCE_VERSION = "1.0"
_WORD = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(frozen=True)
class GovernedDocument:
    """A cleaned document with stable identities used before tokenization."""

    document: TextDocument
    stable_id: str
    exact_cluster_id: str
    near_duplicate_fingerprint: str


@dataclass(frozen=True)
class GovernanceReport:
    """Deterministic cleaning and exact-deduplication counters."""

    input_by_source: dict[str, int]
    accepted_by_source: dict[str, int]
    rejected_by_source_and_reason: dict[str, dict[str, int]]
    exact_duplicate_groups: int
    cross_source_duplicate_groups: int
    near_duplicate_strategy: str


def stable_document_id(document: TextDocument, cleaned_text: str) -> str:
    """Return an identity independent of input order and physical shard."""

    native = document.document_id
    identity = native if native else text_sha256(cleaned_text)
    return hashlib.sha256(
        f"{document.source}\0{identity}".encode("utf-8")
    ).hexdigest()


def near_duplicate_fingerprint(text: str, *, shingle_size: int = 5) -> str:
    """Return a cheap staged fingerprint; it does not claim fuzzy clustering."""

    words = [match.group(0).casefold() for match in _WORD.finditer(text)]
    if len(words) < shingle_size:
        payload = " ".join(words)
    else:
        shingles = {
            " ".join(words[index : index + shingle_size])
            for index in range(len(words) - shingle_size + 1)
        }
        payload = "\n".join(sorted(shingles))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def govern_documents(
    documents: Iterable[TextDocument],
) -> tuple[tuple[GovernedDocument, ...], GovernanceReport]:
    """Normalize and exactly deduplicate within and across all sources.

    Representatives are selected by stable identity, not arrival order. This
    makes the result independent of worker count and Parquet shard traversal.
    """

    input_counts: Counter[str] = Counter()
    rejects: dict[str, Counter[str]] = defaultdict(Counter)
    by_digest: dict[str, list[GovernedDocument]] = defaultdict(list)
    for document in documents:
        input_counts[document.source] += 1
        if not isinstance(document.text, str):
            rejects[document.source]["malformed"] += 1
            continue
        normalized = clean_text(document.text)
        if normalized is None:
            rejects[document.source]["empty_or_invalid"] += 1
            continue
        digest = text_sha256(normalized)
        clean_document = replace(document, text=normalized)
        governed = GovernedDocument(
            document=clean_document,
            stable_id=stable_document_id(clean_document, normalized),
            exact_cluster_id=digest,
            near_duplicate_fingerprint=near_duplicate_fingerprint(normalized),
        )
        by_digest[digest].append(governed)

    accepted: list[GovernedDocument] = []
    exact_groups = 0
    cross_source_groups = 0
    for digest in sorted(by_digest):
        group = sorted(
            by_digest[digest],
            key=lambda item: (item.stable_id, item.document.source),
        )
        accepted.append(group[0])
        if len(group) > 1:
            exact_groups += 1
            sources = {item.document.source for item in group}
            cross_source_groups += int(len(sources) > 1)
            for duplicate in group[1:]:
                reason = (
                    "exact_duplicate_cross_source"
                    if duplicate.document.source != group[0].document.source
                    else "exact_duplicate_within_source"
                )
                rejects[duplicate.document.source][reason] += 1

    accepted.sort(key=lambda item: item.stable_id)
    accepted_counts = Counter(item.document.source for item in accepted)
    sources = sorted(set(input_counts) | set(rejects) | set(accepted_counts))
    report = GovernanceReport(
        input_by_source={source: input_counts[source] for source in sources},
        accepted_by_source={source: accepted_counts[source] for source in sources},
        rejected_by_source_and_reason={
            source: dict(sorted(rejects[source].items())) for source in sources
        },
        exact_duplicate_groups=exact_groups,
        cross_source_duplicate_groups=cross_source_groups,
        near_duplicate_strategy=(
            "SHA-256 word-shingle fingerprints are recorded for staged audits; "
            "full similarity clustering is deferred and must not be reported "
            "as completed."
        ),
    )
    return tuple(accepted), report


def assign_grouped_splits(
    documents: Iterable[GovernedDocument],
    *,
    validation_ratio: float,
    test_ratio: float = 0.0,
    seed: int = 42,
) -> dict[str, tuple[GovernedDocument, ...]]:
    """Assign complete exact-duplicate groups using a stable seeded hash."""

    ratios = (validation_ratio, test_ratio)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
        for value in ratios
    ) or validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio and test_ratio must be non-negative and sum below 1.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer.")

    grouped: dict[str, list[GovernedDocument]] = defaultdict(list)
    for document in documents:
        grouped[document.exact_cluster_id].append(document)
    assigned: dict[str, list[GovernedDocument]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    denominator = float(1 << 256)
    for cluster_id in sorted(grouped):
        value = int.from_bytes(
            hashlib.sha256(f"{seed}\0{cluster_id}".encode("utf-8")).digest(),
            "big",
        ) / denominator
        if value < test_ratio:
            split = "test"
        elif value < test_ratio + validation_ratio:
            split = "validation"
        else:
            split = "train"
        assigned[split].extend(sorted(grouped[cluster_id], key=lambda item: item.stable_id))
    return {name: tuple(values) for name, values in assigned.items()}


def assert_no_split_leakage(
    splits: dict[str, Iterable[GovernedDocument]],
) -> None:
    """Reject stable identities or duplicate clusters occurring in two splits."""

    identity_owner: dict[str, str] = {}
    cluster_owner: dict[str, str] = {}
    for split_name, documents in splits.items():
        for document in documents:
            for key, owners, label in (
                (document.stable_id, identity_owner, "document"),
                (document.exact_cluster_id, cluster_owner, "duplicate cluster"),
            ):
                previous = owners.setdefault(key, split_name)
                if previous != split_name:
                    raise ValueError(
                        f"{label} {key} occurs in both {previous} and {split_name}."
                    )
