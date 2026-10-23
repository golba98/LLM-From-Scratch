"""Dataset preparation statistics."""

from dataclasses import asdict, dataclass
from collections import Counter
from collections.abc import Sequence

from llm_data.data.io import TextDocument


@dataclass(frozen=True)
class DatasetStatistics:
    """Counts, sizes, provenance, and output checksums for one dataset."""

    input_file_count: int
    raw_document_count: int
    cleaned_document_count: int
    empty_documents_removed: int
    duplicate_documents_removed: int
    training_document_count: int
    validation_document_count: int
    total_characters: int
    training_characters: int
    validation_characters: int
    total_utf8_bytes: int
    minimum_document_length: int
    maximum_document_length: int
    mean_document_length: float
    source_counts: dict[str, int]
    train_sha256: str
    validation_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable statistics mapping."""

        return asdict(self)


def build_statistics(
    *,
    input_file_count: int,
    raw_document_count: int,
    cleaned_documents: Sequence[TextDocument],
    empty_documents_removed: int,
    duplicate_documents_removed: int,
    training_documents: Sequence[TextDocument],
    validation_documents: Sequence[TextDocument],
    train_sha256: str,
    validation_sha256: str,
) -> DatasetStatistics:
    """Calculate requested statistics for prepared documents."""

    document_lengths = [len(document.text) for document in cleaned_documents]
    training_characters = sum(
        len(document.text) for document in training_documents
    )
    validation_characters = sum(
        len(document.text) for document in validation_documents
    )
    total_characters = sum(document_lengths)
    total_utf8_bytes = sum(
        len(document.text.encode("utf-8")) for document in cleaned_documents
    )
    source_counts = dict(
        sorted(Counter(document.source for document in cleaned_documents).items())
    )

    return DatasetStatistics(
        input_file_count=input_file_count,
        raw_document_count=raw_document_count,
        cleaned_document_count=len(cleaned_documents),
        empty_documents_removed=empty_documents_removed,
        duplicate_documents_removed=duplicate_documents_removed,
        training_document_count=len(training_documents),
        validation_document_count=len(validation_documents),
        total_characters=total_characters,
        training_characters=training_characters,
        validation_characters=validation_characters,
        total_utf8_bytes=total_utf8_bytes,
        minimum_document_length=min(document_lengths, default=0),
        maximum_document_length=max(document_lengths, default=0),
        mean_document_length=(
            total_characters / len(document_lengths) if document_lengths else 0.0
        ),
        source_counts=source_counts,
        train_sha256=train_sha256,
        validation_sha256=validation_sha256,
    )
