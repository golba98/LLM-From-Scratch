"""Tests for cross-source deduplication and leakage-safe stable splits."""

from src.data.governance import (
    assert_no_split_leakage,
    assign_grouped_splits,
    govern_documents,
)
from src.data.io import TextDocument


def _documents() -> list[TextDocument]:
    return [
        TextDocument("Shared Wikipedia-derived text", "fineweb", "fw-shared"),
        TextDocument("Shared Wikipedia-derived text", "wikipedia", "wiki-shared"),
        TextDocument("FineWeb only", "fineweb", "fw-only"),
        TextDocument("Wikipedia only", "wikipedia", "wiki-only"),
        TextDocument("\x00  ", "fineweb", "empty"),
    ]


def test_cross_source_deduplication_is_order_independent() -> None:
    first, first_report = govern_documents(_documents())
    second, second_report = govern_documents(reversed(_documents()))

    assert first == second
    assert first_report == second_report
    assert len(first) == 3
    assert first_report.cross_source_duplicate_groups == 1
    assert sum(
        reasons.get("exact_duplicate_cross_source", 0)
        for reasons in first_report.rejected_by_source_and_reason.values()
    ) == 1
    assert first_report.rejected_by_source_and_reason["fineweb"][
        "empty_or_invalid"
    ] == 1
    assert "deferred" in first_report.near_duplicate_strategy


def test_group_split_is_stable_and_leakage_safe() -> None:
    governed, _report = govern_documents(_documents())
    first = assign_grouped_splits(
        governed,
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )
    second = assign_grouped_splits(
        reversed(governed),
        validation_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )
    assert first == second
    assert_no_split_leakage(first)
    assert {item.stable_id for values in first.values() for item in values} == {
        item.stable_id for item in governed
    }


def test_leakage_check_rejects_duplicate_cluster() -> None:
    governed, _report = govern_documents(_documents())
    duplicate = governed[0]
    try:
        assert_no_split_leakage(
            {"train": [duplicate], "validation": [duplicate]}
        )
    except ValueError as error:
        assert "occurs in both" in str(error)
    else:
        raise AssertionError("Expected split leakage to be rejected.")
