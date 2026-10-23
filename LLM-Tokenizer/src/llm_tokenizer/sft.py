"""Versioned role-aware SFT serialization with assistant-only labels."""

from dataclasses import dataclass
from typing import Protocol

from llm_tokenizer.tokenizer import (
    ASSISTANT_TOKEN,
    END_TOKEN,
    SYSTEM_TOKEN,
    USER_TOKEN,
)


SFT_FORMAT_VERSION = "chat-v1"
IGNORE_LABEL = -100
ROLE_TOKENS = {
    "system": SYSTEM_TOKEN,
    "user": USER_TOKEN,
    "assistant": ASSISTANT_TOKEN,
}


class ChatTokenizer(Protocol):
    """Minimal tokenizer interface needed by the serializer."""

    def token_to_id(self, token: str) -> int | None: ...

    def encode(self, text: str, add_special_tokens: bool = False): ...


@dataclass(frozen=True)
class ChatMessage:
    """One validated conversation message."""

    role: str
    content: str


@dataclass(frozen=True)
class SerializedConversation:
    """Exact token IDs and assistant-only targets for one conversation."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    supervised_token_count: int
    format_version: str = SFT_FORMAT_VERSION


def format_chat_prompt(
    messages: list[ChatMessage],
    tokenizer: ChatTokenizer,
) -> str:
    """Format completed turns and an assistant turn for inference.

    The final assistant role is intentionally left open: generation begins
    immediately after its role token and must stop at ``END_TOKEN``.
    """

    if not messages:
        raise ValueError("Chat prompt must contain at least one message.")
    if messages[-1].role != "user":
        raise ValueError("Chat prompt must end with a user message.")
    previous: str | None = None
    parts: list[str] = []
    for index, message in enumerate(messages):
        if message.role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported role at message {index}: {message.role!r}.")
        if not message.content.strip():
            raise ValueError(f"Message {index} has empty content.")
        if message.role == "system" and index != 0:
            raise ValueError("System messages are allowed only at the beginning.")
        if index == 0 and message.role == "assistant":
            raise ValueError("A chat prompt cannot begin with an assistant message.")
        if message.role == previous and message.role != "system":
            raise ValueError("Adjacent user or assistant messages are malformed.")
        if previous == "system" and message.role != "user":
            raise ValueError("A system message must be followed by a user message.")
        parts.append(f"{ROLE_TOKENS[message.role]}{message.content}{END_TOKEN}")
        previous = message.role
    parts.append(ASSISTANT_TOKEN)
    return "".join(parts)


def _required_token_id(tokenizer: ChatTokenizer, token: str) -> int:
    token_id = tokenizer.token_to_id(token)
    if token_id is None:
        raise ValueError(f"Tokenizer is missing required chat token {token!r}.")
    return token_id


def serialize_conversation(
    messages: list[ChatMessage],
    tokenizer: ChatTokenizer,
    *,
    maximum_tokens: int | None = None,
) -> SerializedConversation:
    """Serialize a complete conversation and supervise assistant text plus END.

    Conversations are not partially truncated. A conversation over the limit is
    rejected so an assistant response is never trained after being cut in half.
    """

    if not messages:
        raise ValueError("Conversation must contain at least one message.")
    if messages[-1].role != "assistant":
        raise ValueError("Conversation must end with a complete assistant message.")
    previous: str | None = None
    input_ids: list[int] = []
    labels: list[int] = []
    end_id = _required_token_id(tokenizer, END_TOKEN)
    for index, message in enumerate(messages):
        if message.role not in ROLE_TOKENS:
            raise ValueError(f"Unsupported role at message {index}: {message.role!r}.")
        if not message.content.strip():
            raise ValueError(f"Message {index} has empty content.")
        if message.role == "system" and index != 0:
            raise ValueError("System messages are allowed only at the beginning.")
        if index == 0 and message.role == "assistant":
            raise ValueError("A conversation cannot begin with an assistant message.")
        if message.role == previous and message.role != "system":
            raise ValueError("Adjacent user or assistant messages are malformed.")
        if previous == "system" and message.role != "user":
            raise ValueError("A system message must be followed by a user message.")
        role_id = _required_token_id(tokenizer, ROLE_TOKENS[message.role])
        content_ids = list(
            tokenizer.encode(
                message.content,
                add_special_tokens=False,
            ).ids
        )
        segment = [role_id, *content_ids, end_id]
        input_ids.extend(segment)
        if message.role == "assistant":
            labels.extend([IGNORE_LABEL, *content_ids, end_id])
        else:
            labels.extend([IGNORE_LABEL] * len(segment))
        previous = message.role
    if maximum_tokens is not None:
        if maximum_tokens <= 0:
            raise ValueError("maximum_tokens must be positive.")
        if len(input_ids) > maximum_tokens:
            raise ValueError(
                "Conversation exceeds maximum_tokens; discard it instead of "
                "truncating an assistant response."
            )
    supervised = sum(label != IGNORE_LABEL for label in labels)
    if supervised == 0:
        raise ValueError("Conversation contains no supervised assistant tokens.")
    return SerializedConversation(tuple(input_ids), tuple(labels), supervised)
