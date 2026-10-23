"""Deterministic conversation filtering, deduplication, and root-safe splits."""

from collections import Counter
from dataclasses import dataclass
import hashlib
import re

from llm_data.data.cleaning import clean_text
from llm_tokenizer.sft import ChatMessage


_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")
_URL = re.compile(r"https?://\S+", re.IGNORECASE)


@dataclass(frozen=True)
class ConversationCandidate:
    """One reconstructed, complete path and its quality metadata."""

    conversation_id: str
    root_id: str
    messages: tuple[ChatMessage, ...]
    source: str
    review_count: int = 0
    review_result: bool | None = None
    rank: int | None = None
    deleted: bool = False


@dataclass(frozen=True)
class ConversationFilterReport:
    """Accepted count and one primary reason for every rejection."""

    input_count: int
    accepted_count: int
    rejected_by_reason: dict[str, int]
    exact_duplicate_groups: int


def _content_digest(messages: tuple[ChatMessage, ...]) -> str:
    payload = "\n".join(
        f"{message.role}\0{message.content}" for message in messages
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rejection_reason(
    candidate: ConversationCandidate,
    *,
    minimum_review_count: int,
    maximum_rank: int | None,
) -> str | None:
    if candidate.deleted:
        return "deleted"
    if not candidate.root_id or not candidate.conversation_id:
        return "missing_identity"
    if len(candidate.messages) < 2:
        return "incomplete_path"
    if candidate.messages[0].role not in {"system", "user"}:
        return "malformed_roles"
    if candidate.messages[-1].role != "assistant":
        return "incomplete_path"
    previous: str | None = None
    for index, message in enumerate(candidate.messages):
        cleaned = clean_text(message.content)
        if cleaned is None:
            return "empty_message"
        if message.role not in {"system", "user", "assistant"}:
            return "malformed_roles"
        if message.role == "system" and index != 0:
            return "malformed_roles"
        if previous == message.role and message.role != "system":
            return "malformed_roles"
        previous = message.role
        if _EMAIL.search(cleaned) or _PHONE.search(cleaned):
            return "clear_pii"
        if len(_URL.findall(cleaned)) >= 4:
            return "spam"
    assistant_texts = [
        message.content.strip()
        for message in candidate.messages
        if message.role == "assistant"
    ]
    if any(len(text) < 2 for text in assistant_texts):
        return "defective_assistant_answer"
    if candidate.review_result is False:
        return "failed_review"
    if candidate.review_count < minimum_review_count:
        return "insufficient_reviews"
    if maximum_rank is not None and candidate.rank is not None and candidate.rank > maximum_rank:
        return "rank_below_threshold"
    return None


def filter_conversations(
    candidates: list[ConversationCandidate],
    *,
    minimum_review_count: int = 0,
    maximum_rank: int | None = None,
) -> tuple[tuple[ConversationCandidate, ...], ConversationFilterReport]:
    """Filter structurally and by metadata without deleting safe refusals."""

    if minimum_review_count < 0:
        raise ValueError("minimum_review_count must be non-negative.")
    rejected: Counter[str] = Counter()
    by_digest: dict[str, list[ConversationCandidate]] = {}
    for candidate in candidates:
        reason = _rejection_reason(
            candidate,
            minimum_review_count=minimum_review_count,
            maximum_rank=maximum_rank,
        )
        if reason is not None:
            rejected[reason] += 1
            continue
        digest = _content_digest(candidate.messages)
        by_digest.setdefault(digest, []).append(candidate)
    accepted: list[ConversationCandidate] = []
    duplicate_groups = 0
    for digest in sorted(by_digest):
        group = sorted(
            by_digest[digest],
            key=lambda item: (item.root_id, item.conversation_id),
        )
        accepted.append(group[0])
        if len(group) > 1:
            duplicate_groups += 1
            rejected["exact_duplicate_conversation"] += len(group) - 1
    accepted.sort(key=lambda item: (item.root_id, item.conversation_id))
    return tuple(accepted), ConversationFilterReport(
        input_count=len(candidates),
        accepted_count=len(accepted),
        rejected_by_reason=dict(sorted(rejected.items())),
        exact_duplicate_groups=duplicate_groups,
    )


def split_conversations_by_root(
    candidates: list[ConversationCandidate] | tuple[ConversationCandidate, ...],
    *,
    validation_ratio: float,
    seed: int = 42,
) -> dict[str, tuple[ConversationCandidate, ...]]:
    """Assign complete root trees so overlapping paths cannot cross splits."""

    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must satisfy 0 <= ratio < 1.")
    root_splits: dict[str, str] = {}
    denominator = float(1 << 256)
    for root_id in {candidate.root_id for candidate in candidates}:
        value = int.from_bytes(
            hashlib.sha256(f"{seed}\0{root_id}".encode()).digest(),
            "big",
        ) / denominator
        root_splits[root_id] = "validation" if value < validation_ratio else "train"
    values: dict[str, list[ConversationCandidate]] = {"train": [], "validation": []}
    for candidate in sorted(candidates, key=lambda item: (item.root_id, item.conversation_id)):
        values[root_splits[candidate.root_id]].append(candidate)
    return {name: tuple(items) for name, items in values.items()}
