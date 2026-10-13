"""Assertion-based tests for generated-text evaluation helpers."""

import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.evaluate_checkpoint import (
    _build_prompt,
    _expected_term_matches,
    _read_prompts,
    _validate_tokenizer_compatibility,
)
from src.evaluation import (
    analyze_generated_text,
    ngram_overlap_rate,
    perplexity_from_loss,
    repeated_ngram_rate,
)


class _WhitespaceTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool):
        assert not add_special_tokens
        return SimpleNamespace(ids=list(range(1, len(text.split()) + 1)))

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(f"token-{token_id}" for token_id in token_ids)


def _raises(exception_type: type[BaseException], function) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"Expected {exception_type.__name__}.")


def main() -> None:
    assert math.isclose(perplexity_from_loss(math.log(8)), 8.0)
    _raises(ValueError, lambda: perplexity_from_loss(-1))
    _raises(ValueError, lambda: perplexity_from_loss(float("nan")))

    words = "one two three four one two three four".split()
    assert repeated_ngram_rate(words, 4) > 0
    assert repeated_ngram_rate(["short"], 4) == 0
    _raises(ValueError, lambda: repeated_ngram_rate(words, 0))

    clean = analyze_generated_text("A small bird flew over the green tree.")
    assert clean.word_count == 8
    assert clean.malformed_character_count == 0
    assert clean.longest_repeated_run == 1
    assert json.loads(json.dumps(clean.to_dict())) == clean.to_dict()

    malformed = analyze_generated_text("bad\ufffd\x00 text word word word")
    assert malformed.malformed_character_count == 2
    assert malformed.longest_repeated_run == 3

    overlap = ngram_overlap_rate(
        "the little cat sat on the warm red mat",
        "yesterday the little cat sat on the warm red mat quietly",
        ngram_size=4,
    )
    assert overlap > 0.5
    assert ngram_overlap_rate("too short", "too short", ngram_size=4) == 0

    prompts = _read_prompts(Path("configs/evaluation_prompts.json"))
    assert len(prompts) >= 10
    long_context = next(
        prompt
        for prompt in prompts
        if prompt["category"] == "long_context_retention"
    )
    rendered, token_ids = _build_prompt(
        long_context,
        tokenizer=_WhitespaceTokenizer(),
        context_length=2048,
    )
    assert rendered
    assert len(token_ids) == 1536
    assert _expected_term_matches(
        ["sun"],
        "wrong immediate answer before the sun appears later",
        search_characters=20,
    ) == {"sun": False}
    assert _expected_term_matches(
        ["sun"],
        "sun is the immediate answer",
        search_characters=20,
    ) == {"sun": True}
    _raises(
        ValueError,
        lambda: _build_prompt(
            long_context,
            tokenizer=_WhitespaceTokenizer(),
            context_length=1024,
        ),
    )
    _validate_tokenizer_compatibility(
        checkpoint_checksum="abc",
        tokenizer_checksum="abc",
        tokenizer_vocab_size=300,
        model_vocab_size=512,
    )
    _raises(
        ValueError,
        lambda: _validate_tokenizer_compatibility(
            checkpoint_checksum="abc",
            tokenizer_checksum="wrong",
            tokenizer_vocab_size=300,
            model_vocab_size=512,
        ),
    )
    _raises(
        ValueError,
        lambda: _validate_tokenizer_compatibility(
            checkpoint_checksum=None,
            tokenizer_checksum="abc",
            tokenizer_vocab_size=513,
            model_vocab_size=512,
        ),
    )

    print("All evaluation-helper tests passed.")


if __name__ == "__main__":
    main()
