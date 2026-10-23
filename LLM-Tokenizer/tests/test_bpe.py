"""Public iterable training and fixed special-token contracts."""

from llm_tokenizer.tokenizer import SPECIAL_TOKENS, train_bpe


def test_iterable_training_and_unicode_round_trip():
    texts = ["Hello café 👋", "Tokenizer byte-level round trip", "Hello café 👋"]
    tokenizer = train_bpe(iter(texts), vocab_size=300, min_frequency=1)
    assert [tokenizer.token_to_id(t) for t in SPECIAL_TOKENS] == list(range(8))
    for text in texts:
        assert tokenizer.decode(tokenizer.encode(text).ids) == text
