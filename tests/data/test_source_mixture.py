"""Tests for deterministic source-mixture scheduling and accounting."""

from src.data.mixture import DocumentBudget, build_mixture_schedule


def _documents() -> list[DocumentBudget]:
    return [
        DocumentBudget("fineweb", f"f-{index}", 100 + index)
        for index in range(8)
    ] + [
        DocumentBudget("wikipedia", f"w-{index}", 80 + index)
        for index in range(6)
    ]


def test_mixture_is_deterministic_interleaved_and_accounted() -> None:
    first, report = build_mixture_schedule(
        _documents(),
        requested_percentages={"fineweb": 0.8, "wikipedia": 0.2},
        total_token_budget=900,
        seed=42,
    )
    second, repeated_report = build_mixture_schedule(
        list(reversed(_documents())),
        requested_percentages={"wikipedia": 20, "fineweb": 80},
        total_token_budget=900,
        seed=42,
    )
    assert first == second
    assert report == repeated_report
    assert report.achieved_total_tokens == 900
    assert sum(report.tokens_by_source.values()) == 900
    assert len({item.document_id for item in first}) == len(first)
    assert report.repeated_documents == 0
    assert {item.source for item in first[:3]} == {"fineweb", "wikipedia"}
    assert sum(item.truncated_tokens for item in first) == sum(
        report.truncated_tokens_by_source.values()
    )


def test_mixture_reports_source_exhaustion_without_oversampling() -> None:
    documents = [
        DocumentBudget("fineweb", "f", 100),
        DocumentBudget("wikipedia", "w", 10),
    ]
    selected, report = build_mixture_schedule(
        documents,
        requested_percentages={"fineweb": 0.5, "wikipedia": 0.5},
        total_token_budget=500,
    )
    assert len(selected) == 2
    assert report.achieved_total_tokens == 110
    assert report.repeated_documents == 0
    assert set(report.exhausted_sources) == {"fineweb", "wikipedia"}
