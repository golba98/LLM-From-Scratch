"""JSONL convenience adapters; the BPE engine accepts plain text iterables."""
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass(frozen=True)
class TokenizerDocument:
    text: str
    source: str
    document_id: str | None = None
    metadata: dict | None = None

def load_documents(paths):
    documents = []
    for path in map(Path, paths):
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                        raise ValueError("required field 'text' must be a string")
                    source = row.get("source", path.name)
                    if not isinstance(source, str):
                        raise ValueError("optional field 'source' must be a string")
                    identity = row.get("document_id")
                    metadata = row.get("metadata")
                    if identity is not None and not isinstance(identity, str):
                        raise ValueError("optional field 'document_id' must be a string or null")
                    if metadata is not None and not isinstance(metadata, dict):
                        raise ValueError("optional field 'metadata' must be an object or null")
                    documents.append(TokenizerDocument(row["text"], source, identity, metadata))
                except (ValueError, TypeError) as error:
                    raise ValueError(f"{path}:{number}: {error}") from error
    return documents

def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
