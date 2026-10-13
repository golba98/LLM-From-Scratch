"""Deterministic, dependency-free metrics for generated-text evaluation."""

from dataclasses import asdict, dataclass
import math
import re
import unicodedata


_WORD_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


@dataclass(frozen=True)
class TextQualityMetrics:
    """Mechanical quality signals for one generated sample."""

    character_count: int
    word_count: int
    unique_word_ratio: float
    repeated_ngram_rate: float
    malformed_character_count: int
    longest_repeated_run: int
    longest_repeated_span: int
    distinct_2: float
    distinct_3: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def perplexity_from_loss(loss: float) -> float:
    """Convert finite mean cross-entropy to perplexity."""

    if not isinstance(loss, (int, float)) or isinstance(loss, bool):
        raise ValueError("loss must be a finite non-negative number.")
    value = float(loss)
    if not math.isfinite(value) or value < 0:
        raise ValueError("loss must be a finite non-negative number.")
    try:
        return math.exp(value)
    except OverflowError:
        return math.inf


def _tokens(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _WORD_PATTERN.finditer(text)]


def repeated_ngram_rate(words: list[str], ngram_size: int = 4) -> float:
    """Return the fraction of n-gram occurrences beyond their first use."""

    if (
        not isinstance(ngram_size, int)
        or isinstance(ngram_size, bool)
        or ngram_size <= 0
    ):
        raise ValueError("ngram_size must be a positive integer.")
    count = len(words) - ngram_size + 1
    if count <= 0:
        return 0.0
    ngrams = [
        tuple(words[index : index + ngram_size])
        for index in range(count)
    ]
    return 1.0 - (len(set(ngrams)) / len(ngrams))


def distinct_ngram_ratio(words: list[str], ngram_size: int) -> float:
    """Return unique n-grams divided by all observed n-grams."""

    count = len(words) - ngram_size + 1
    if count <= 0:
        return 0.0
    values = {
        tuple(words[index : index + ngram_size]) for index in range(count)
    }
    return len(values) / count


def longest_repeated_span(words: list[str]) -> int:
    """Return the longest repeated non-overlapping word span."""

    longest = 0
    for left in range(len(words)):
        for right in range(left + 1, len(words)):
            length = 0
            while (
                right + length < len(words)
                and left + length < right
                and words[left + length] == words[right + length]
            ):
                length += 1
            longest = max(longest, length)
    return longest


def ngram_overlap_rate(
    generated_text: str,
    reference_text: str,
    *,
    ngram_size: int = 8,
) -> float:
    """Measure generated n-grams that also occur in reference material."""

    generated_words = _tokens(generated_text)
    reference_words = _tokens(reference_text)
    if len(generated_words) < ngram_size:
        return 0.0
    reference_ngrams = {
        tuple(reference_words[index : index + ngram_size])
        for index in range(len(reference_words) - ngram_size + 1)
    }
    generated_ngrams = [
        tuple(generated_words[index : index + ngram_size])
        for index in range(len(generated_words) - ngram_size + 1)
    ]
    if not generated_ngrams:
        return 0.0
    matches = sum(ngram in reference_ngrams for ngram in generated_ngrams)
    return matches / len(generated_ngrams)


def analyze_generated_text(
    text: str,
    *,
    ngram_size: int = 4,
) -> TextQualityMetrics:
    """Calculate mechanical quality metrics without subjective scoring."""

    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    words = _tokens(text)
    malformed_count = sum(
        character == "\ufffd"
        or (
            unicodedata.category(character) == "Cc"
            and character not in "\n\r\t"
        )
        for character in text
    )
    longest_run = 0
    current_run = 0
    previous: str | None = None
    for word in words:
        current_run = current_run + 1 if word == previous else 1
        longest_run = max(longest_run, current_run)
        previous = word
    return TextQualityMetrics(
        character_count=len(text),
        word_count=len(words),
        unique_word_ratio=(
            len(set(words)) / len(words) if words else 0.0
        ),
        repeated_ngram_rate=repeated_ngram_rate(words, ngram_size),
        malformed_character_count=malformed_count,
        longest_repeated_run=longest_run,
        longest_repeated_span=longest_repeated_span(words),
        distinct_2=distinct_ngram_ratio(words, 2),
        distinct_3=distinct_ngram_ratio(words, 3),
    )
