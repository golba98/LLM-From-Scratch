"""Executable unit tests for dataset cleaning and splitting."""

from collections.abc import Callable

from src.data.cleaning import clean_and_deduplicate, clean_text, text_sha256
from src.data.io import TextDocument
from src.data.split import split_documents


def assert_raises(
    exception_type: type[BaseException],
    operation: Callable[[], object],
    message_fragment: str,
) -> None:
    """Assert that an operation raises an informative exception."""

    try:
        operation()
    except exception_type as error:
        assert message_fragment in str(error)
    else:
        raise AssertionError(f"Expected {exception_type.__name__} to be raised.")


def test_clean_text() -> None:
    """Apply every required normalization without changing language."""

    raw_text = (
        "  Ｈｅｌｌｏ\u00a0\tworld!!!  \r\n"
        "Line\x00\x01  two   \r"
        "Old\tMac\n\n\n\nFinal?  "
    )
    assert clean_text(raw_text) == (
        "Hello world!!!\nLine two\nOld Mac\n\nFinal?"
    )
    assert clean_text("  \x00\t\n\r  ") is None
    assert clean_text("Keep MIXED Case, punctuation: yes!") == (
        "Keep MIXED Case, punctuation: yes!"
    )


def test_exact_deduplication() -> None:
    """Keep the first cleaned exact duplicate in stable input order."""

    documents = [
        TextDocument(
            text="Alpha   text",
            source="first.txt",
            document_id="first",
            metadata={"order": 1},
        ),
        TextDocument(text="Alpha text", source="duplicate.txt"),
        TextDocument(text="\x00 \t\n", source="empty.txt"),
        TextDocument(text="Beta", source="last.txt"),
    ]
    result = clean_and_deduplicate(documents)

    assert [document.text for document in result.documents] == [
        "Alpha text",
        "Beta",
    ]
    assert result.documents[0].source == "first.txt"
    assert result.documents[0].document_id == "first"
    assert result.documents[0].metadata == {"order": 1}
    assert result.empty_documents_removed == 1
    assert result.duplicate_documents_removed == 1
    assert text_sha256("Alpha text") == (
        "43760b534228833c83cbd58895b35c130a1750910390484cfa9f067f6f8f4491"
    )


def test_deterministic_split() -> None:
    """Split by SHA-256 deterministically and preserve order per split."""

    documents = [
        TextDocument(text=f"Document {index}", source="sample")
        for index in range(6)
    ]
    first_train, first_validation = split_documents(
        documents,
        validation_ratio=0.25,
        seed=42,
    )
    second_train, second_validation = split_documents(
        documents,
        validation_ratio=0.25,
        seed=42,
    )
    assert first_train == second_train
    assert first_validation == second_validation

    original_positions = {document.text: index for index, document in enumerate(documents)}
    assert [original_positions[document.text] for document in first_train] == sorted(
        original_positions[document.text] for document in first_train
    )
    assert [
        original_positions[document.text] for document in first_validation
    ] == sorted(
        original_positions[document.text] for document in first_validation
    )

    all_training, no_validation = split_documents(
        documents,
        validation_ratio=0.0,
    )
    assert all_training == documents
    assert no_validation == []

    _, forced_validation = split_documents(
        documents[:2],
        validation_ratio=1e-12,
        seed=42,
    )
    assert len(forced_validation) == 1

    for invalid_ratio in (-0.1, 1.0, float("inf"), True):
        assert_raises(
            ValueError,
            lambda invalid_ratio=invalid_ratio: split_documents(
                documents,
                validation_ratio=invalid_ratio,
            ),
            "validation_ratio must satisfy 0 <= ratio < 1",
        )


def main() -> None:
    """Run all cleaning and splitting tests."""

    test_clean_text()
    test_exact_deduplication()
    test_deterministic_split()
    print("All data cleaning tests passed.")


if __name__ == "__main__":
    main()
