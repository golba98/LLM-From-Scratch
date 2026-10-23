"""Deterministic document-level dataset splitting."""

import hashlib
import math
from collections.abc import Sequence

from llm_data.data.io import TextDocument


def _assignment_value(document: TextDocument, seed: int) -> int:
    identity = document.document_id or document.text
    payload = f"{seed}\0{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big")


def split_documents(
    documents: Sequence[TextDocument],
    *,
    validation_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[TextDocument], list[TextDocument]]:
    """Split documents deterministically while preserving relative order."""

    if (
        not isinstance(validation_ratio, (int, float))
        or isinstance(validation_ratio, bool)
        or not math.isfinite(float(validation_ratio))
        or not 0.0 <= float(validation_ratio) < 1.0
    ):
        raise ValueError(
            "validation_ratio must satisfy 0 <= ratio < 1; "
            f"got {validation_ratio!r}."
        )
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer; got {seed!r}.")

    ratio = float(validation_ratio)
    assignment_values = [
        _assignment_value(document, seed) for document in documents
    ]
    threshold = int(ratio * (1 << 256))
    validation_indices = {
        index
        for index, value in enumerate(assignment_values)
        if value < threshold
    }

    if ratio > 0.0 and len(documents) >= 2 and not validation_indices:
        validation_indices.add(
            min(
                range(len(documents)),
                key=assignment_values.__getitem__,
            )
        )

    training_documents = [
        document
        for index, document in enumerate(documents)
        if index not in validation_indices
    ]
    validation_documents = [
        document
        for index, document in enumerate(documents)
        if index in validation_indices
    ]
    return training_documents, validation_documents
