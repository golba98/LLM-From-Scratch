"""Exact-token tests for versioned assistant-only SFT serialization."""

from types import SimpleNamespace

from src.sft import ChatMessage, IGNORE_LABEL, format_chat_prompt, serialize_conversation
from src.tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
)


class FakeTokenizer:
    """Small deterministic tokenizer used to inspect exact masks."""

    tokens = {
        SYSTEM_TOKEN: 4,
        USER_TOKEN: 5,
        ASSISTANT_TOKEN: 6,
        END_TOKEN: 7,
    }

    def token_to_id(self, token: str) -> int | None:
        return self.tokens.get(token)

    def encode(self, text: str, add_special_tokens: bool = False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[100 + ord(character) for character in text])


def test_exact_role_tokens_and_assistant_mask() -> None:
    tokenizer = FakeTokenizer()
    serialized = serialize_conversation(
        [
            ChatMessage("system", "S"),
            ChatMessage("user", "Q"),
            ChatMessage("assistant", "A"),
        ],
        tokenizer,
    )
    assert serialized.input_ids == (4, 183, 7, 5, 181, 7, 6, 165, 7)
    assert serialized.labels == (
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        165,
        7,
    )
    assert serialized.supervised_token_count == 2


def test_multi_turn_and_invalid_sequences() -> None:
    tokenizer = FakeTokenizer()
    serialized = serialize_conversation(
        [
            ChatMessage("user", "a"),
            ChatMessage("assistant", "b"),
            ChatMessage("user", "c"),
            ChatMessage("assistant", "d"),
        ],
        tokenizer,
    )
    assert serialized.supervised_token_count == 4
    assert serialized.labels.count(7) == 2

    invalid = (
        [ChatMessage("user", "unfinished")],
        [ChatMessage("assistant", "orphan")],
        [ChatMessage("user", "one"), ChatMessage("user", "two")],
    )
    for messages in invalid:
        try:
            serialize_conversation(messages, tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError("Expected malformed conversation rejection.")


def test_chat_prompt_leaves_assistant_turn_open() -> None:
    prompt = format_chat_prompt(
        [ChatMessage("user", "hello"), ChatMessage("assistant", "reply"), ChatMessage("user", "again")],
        FakeTokenizer(),
    )
    assert prompt == "<|user|>hello<|end|><|assistant|>reply<|end|><|user|>again<|end|><|assistant|>"

    try:
        format_chat_prompt([ChatMessage("assistant", "orphan")], FakeTokenizer())
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a prompt to end with a user message.")


def test_incomplete_assistant_is_not_truncated() -> None:
    try:
        serialize_conversation(
            [ChatMessage("user", "q"), ChatMessage("assistant", "answer")],
            FakeTokenizer(),
            maximum_tokens=5,
        )
    except ValueError as error:
        assert "discard it" in str(error)
    else:
        raise AssertionError("Expected overlength conversation rejection.")


def test_next_token_shift_targets_assistant_content_and_end() -> None:
    """The trainer shift must open supervision after the assistant role."""

    tokenizer = FakeTokenizer()
    serialized = serialize_conversation(
        [
            ChatMessage("user", "Q"),
            ChatMessage("assistant", "A"),
        ],
        tokenizer,
    )
    shifted_labels = list(serialized.labels[1:]) + [IGNORE_LABEL]
    # The assistant role is input position 3; its next-token target is A.
    assert serialized.input_ids == (5, 181, 7, 6, 165, 7)
    assert shifted_labels == [
        IGNORE_LABEL,
        IGNORE_LABEL,
        IGNORE_LABEL,
        165,
        7,
        IGNORE_LABEL,
    ]
