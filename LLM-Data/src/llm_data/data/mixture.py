"""Deterministic token-budget source interleaving and accounting."""

from dataclasses import asdict, dataclass
import hashlib
import math


@dataclass(frozen=True)
class DocumentBudget:
    """Post-tokenization size for one unique source document."""

    source: str
    document_id: str
    token_count: int

    def __post_init__(self) -> None:
        if not self.source or not self.document_id:
            raise ValueError("source and document_id must be non-empty.")
        if self.token_count <= 0:
            raise ValueError("token_count must be positive.")


@dataclass(frozen=True)
class MixtureSelection:
    """One selected document and its contributed token budget."""

    source: str
    document_id: str
    available_tokens: int
    contributed_tokens: int
    truncated_tokens: int


@dataclass(frozen=True)
class MixtureReport:
    """Requested and achieved source accounting for one schedule."""

    seed: int
    requested_total_tokens: int
    achieved_total_tokens: int
    requested_percentages: dict[str, float]
    achieved_percentages: dict[str, float]
    documents_by_source: dict[str, int]
    tokens_by_source: dict[str, int]
    truncated_tokens_by_source: dict[str, int]
    repeated_documents: int
    exhausted_sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _stable_order(document: DocumentBudget, seed: int) -> bytes:
    return hashlib.sha256(
        f"{seed}\0{document.source}\0{document.document_id}".encode("utf-8")
    ).digest()


def build_mixture_schedule(
    documents: list[DocumentBudget],
    *,
    requested_percentages: dict[str, float],
    total_token_budget: int,
    seed: int = 42,
) -> tuple[tuple[MixtureSelection, ...], MixtureReport]:
    """Interleave unique documents toward requested token shares.

    The final selected document may be truncated to hit the global budget.
    Documents are never repeated or oversampled by this initial policy.
    """

    if total_token_budget <= 0:
        raise ValueError("total_token_budget must be positive.")
    if not requested_percentages:
        raise ValueError("requested_percentages must not be empty.")
    if set(requested_percentages) != {document.source for document in documents}:
        raise ValueError("requested_percentages must exactly match document sources.")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
        for value in requested_percentages.values()
    ):
        raise ValueError("Every requested percentage must be positive and finite.")
    total_weight = sum(float(value) for value in requested_percentages.values())
    normalized = {
        source: float(value) / total_weight
        for source, value in sorted(requested_percentages.items())
    }
    ids = [document.document_id for document in documents]
    if len(ids) != len(set(ids)):
        raise ValueError("document_id values must be globally unique.")

    queues: dict[str, list[DocumentBudget]] = {
        source: sorted(
            (document for document in documents if document.source == source),
            key=lambda document: _stable_order(document, seed),
        )
        for source in normalized
    }
    cursors = {source: 0 for source in normalized}
    contributed = {source: 0 for source in normalized}
    truncated = {source: 0 for source in normalized}
    document_counts = {source: 0 for source in normalized}
    selections: list[MixtureSelection] = []

    while sum(contributed.values()) < total_token_budget:
        available = [
            source
            for source in normalized
            if cursors[source] < len(queues[source])
        ]
        if not available:
            break
        processed = sum(contributed.values())
        source = max(
            available,
            key=lambda name: (
                normalized[name] * max(processed, 1) - contributed[name],
                normalized[name],
                name,
            ),
        )
        document = queues[source][cursors[source]]
        cursors[source] += 1
        remaining = total_token_budget - processed
        contribution = min(document.token_count, remaining)
        discarded = document.token_count - contribution
        selections.append(
            MixtureSelection(
                source=source,
                document_id=document.document_id,
                available_tokens=document.token_count,
                contributed_tokens=contribution,
                truncated_tokens=discarded,
            )
        )
        contributed[source] += contribution
        truncated[source] += discarded
        document_counts[source] += 1

    achieved_total = sum(contributed.values())
    achieved = {
        source: (
            contributed[source] / achieved_total if achieved_total else 0.0
        )
        for source in normalized
    }
    exhausted = tuple(
        source
        for source in normalized
        if cursors[source] == len(queues[source])
    )
    report = MixtureReport(
        seed=seed,
        requested_total_tokens=total_token_budget,
        achieved_total_tokens=achieved_total,
        requested_percentages=normalized,
        achieved_percentages=achieved,
        documents_by_source=document_counts,
        tokens_by_source=contributed,
        truncated_tokens_by_source=truncated,
        repeated_documents=0,
        exhausted_sources=exhausted,
    )
    return tuple(selections), report
