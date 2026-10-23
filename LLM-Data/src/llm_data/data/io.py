"""Input and output helpers for text dataset documents."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class TextDocument:
    """One source document before or after cleaning."""

    text: str
    source: str
    document_id: str | None = None
    metadata: dict[str, object] | None = None


def _read_text_file(
    path: Path,
    *,
    split_text_on_blank_lines: bool,
) -> list[TextDocument]:
    text = path.read_text(encoding="utf-8")
    if not split_text_on_blank_lines:
        return [TextDocument(text=text, source=path.name)]

    normalized_line_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(
        r"\n[ \t]*\n(?:(?:[ \t]*\n)*)",
        normalized_line_endings,
    )
    return [TextDocument(text=part, source=path.name) for part in parts]


def _jsonl_error(path: Path, line_number: int, message: str) -> ValueError:
    return ValueError(f"{path}:{line_number}: {message}")


def _read_jsonl_file(path: Path) -> list[TextDocument]:
    documents: list[TextDocument] = []
    with path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise _jsonl_error(
                    path,
                    line_number,
                    f"malformed JSON ({error.msg})",
                ) from error

            if not isinstance(record, dict):
                raise _jsonl_error(
                    path,
                    line_number,
                    "each JSONL row must be an object",
                )

            text = record.get("text")
            if not isinstance(text, str):
                raise _jsonl_error(
                    path,
                    line_number,
                    "required field 'text' must be a string",
                )

            source = record.get("source", path.name)
            if not isinstance(source, str):
                raise _jsonl_error(
                    path,
                    line_number,
                    "optional field 'source' must be a string",
                )

            document_id = record.get("document_id")
            if document_id is not None and not isinstance(document_id, str):
                raise _jsonl_error(
                    path,
                    line_number,
                    "optional field 'document_id' must be a string or null",
                )

            metadata = record.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise _jsonl_error(
                    path,
                    line_number,
                    "optional field 'metadata' must be an object or null",
                )

            documents.append(
                TextDocument(
                    text=text,
                    source=source,
                    document_id=document_id,
                    metadata=metadata,
                )
            )
    return documents


def read_documents(
    path: str | Path,
    *,
    split_text_on_blank_lines: bool = False,
) -> list[TextDocument]:
    """Read all documents from one UTF-8 text or JSONL/NDJSON file."""

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".txt":
        return _read_text_file(
            input_path,
            split_text_on_blank_lines=split_text_on_blank_lines,
        )
    if suffix in {".jsonl", ".ndjson"}:
        return _read_jsonl_file(input_path)
    raise ValueError(
        f"Unsupported input format for {input_path}; "
        "expected .txt, .jsonl, or .ndjson."
    )


def load_documents(
    paths: Sequence[str | Path],
    *,
    split_text_on_blank_lines: bool = False,
) -> list[TextDocument]:
    """Read documents from input files in stable supplied order."""

    if not paths:
        raise ValueError("At least one input path is required.")

    documents: list[TextDocument] = []
    for path in paths:
        documents.extend(
            read_documents(
                path,
                split_text_on_blank_lines=split_text_on_blank_lines,
            )
        )
    return documents


def document_to_record(document: TextDocument) -> dict[str, object]:
    """Convert a document to its clean JSONL representation."""

    record: dict[str, object] = {
        "text": document.text,
        "source": document.source,
    }
    if document.document_id is not None:
        record["document_id"] = document.document_id
    if document.metadata is not None:
        record["metadata"] = document.metadata
    return record


def write_jsonl(
    path: str | Path,
    documents: Iterable[TextDocument],
) -> str:
    """Write deterministic JSONL and return its SHA-256 checksum."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()

    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for document in documents:
            line = (
                json.dumps(
                    document_to_record(document),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            output_file.write(line)
            digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def write_json(path: str | Path, data: object) -> None:
    """Write stable, human-readable UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        data,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    output_path.write_text(serialized + "\n", encoding="utf-8")
