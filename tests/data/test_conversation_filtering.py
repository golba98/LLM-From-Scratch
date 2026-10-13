"""Tests for refusal-preserving filters and tree-safe conversation splits."""

from src.data.conversation import (
    ConversationCandidate,
    filter_conversations,
    split_conversations_by_root,
)
from src.sft import ChatMessage


def _candidate(identifier: str, root: str, answer: str) -> ConversationCandidate:
    return ConversationCandidate(
        conversation_id=identifier,
        root_id=root,
        source="oasst1",
        messages=(ChatMessage("user", "How should I respond?"), ChatMessage("assistant", answer)),
        review_count=2,
        review_result=True,
        rank=0,
    )


def test_filter_keeps_good_refusals_and_records_reasons() -> None:
    refusal = _candidate(
        "refusal",
        "root-a",
        "I cannot help with harm, but I can suggest a safe alternative.",
    )
    pii = _candidate("pii", "root-b", "Email jane@example.com for the answer.")
    deleted = ConversationCandidate(
        **{**_candidate("deleted", "root-c", "A valid answer").__dict__, "deleted": True}
    )
    duplicate = _candidate("duplicate", "root-d", refusal.messages[-1].content)
    accepted, report = filter_conversations([refusal, pii, deleted, duplicate])
    assert [item.conversation_id for item in accepted] == ["refusal"]
    assert report.rejected_by_reason == {
        "clear_pii": 1,
        "deleted": 1,
        "exact_duplicate_conversation": 1,
    }


def test_root_paths_never_cross_splits() -> None:
    paths = [
        _candidate("a-1", "same-root", "First branch"),
        _candidate("a-2", "same-root", "Second branch"),
        _candidate("b-1", "other-root", "Other answer"),
    ]
    splits = split_conversations_by_root(paths, validation_ratio=0.5, seed=42)
    owners = {
        item.root_id: split
        for split, items in splits.items()
        for item in items
    }
    assert len(owners) == 2
    assert sum(len(items) for items in splits.values()) == 3
