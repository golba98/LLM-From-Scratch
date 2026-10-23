"""Text normalization and exact document deduplication."""

from dataclasses import dataclass, replace
import hashlib
import re
import unicodedata
from collections.abc import Iterable

from llm_data.data.io import TextDocument


CLEANING_VERSION = "1.0"


@dataclass(frozen=True)
class CleaningResult:
    """Clean documents and counters for removed records."""

    documents: tuple[TextDocument, ...]
    empty_documents_removed: int
    duplicate_documents_removed: int


def clean_text(text: str) -> str | None:
    """Normalize one document while preserving its language and punctuation."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.replace("\x00", "").replace("\u00a0", " ")
    normalized = "".join(
        character
        for character in normalized
        if character in {"\t", "\n"}
        or unicodedata.category(character) != "Cc"
    )

    cleaned_lines = []
    for line in normalized.split("\n"):
        collapsed = re.sub(r"[^\S\n]+", " ", line)
        cleaned_lines.append(collapsed.rstrip())

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = cleaned.strip()
    return cleaned or None


def text_sha256(text: str) -> str:
    """Return the SHA-256 digest for cleaned UTF-8 text."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_and_deduplicate(
    documents: Iterable[TextDocument],
) -> CleaningResult:
    """Clean documents, remove empties, and keep the first exact duplicate."""

    cleaned_documents: list[TextDocument] = []
    seen_digests: set[str] = set()
    empty_removed = 0
    duplicates_removed = 0

    for document in documents:
        cleaned_text = clean_text(document.text)
        if cleaned_text is None:
            empty_removed += 1
            continue

        digest = text_sha256(cleaned_text)
        if digest in seen_digests:
            duplicates_removed += 1
            continue

        seen_digests.add(digest)
        cleaned_documents.append(replace(document, text=cleaned_text))

    return CleaningResult(
        documents=tuple(cleaned_documents),
        empty_documents_removed=empty_removed,
        duplicate_documents_removed=duplicates_removed,
    )
