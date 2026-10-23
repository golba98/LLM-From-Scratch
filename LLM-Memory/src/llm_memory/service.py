"""Opt-in memory lifecycle and graceful failure handling for native chat."""

from dataclasses import asdict
from pathlib import Path
import time
import uuid

from llm_memory.client import EncoderClient
from llm_memory.store import MemoryHit, MemoryStore


class ConversationMemory:
    """Keep user scopes explicit and persistence disabled unless selected."""

    def __init__(self, *, user_id: str = "local", mode: str = "off",
                 path: Path = Path("data/memory/codexa.sqlite3"),
                 device: str = "cpu", threshold: float = 0.68,
                 client=None) -> None:
        if not user_id or not -1 <= threshold <= 1:
            raise ValueError("Invalid user identity or retrieval threshold.")
        self.user_id = user_id
        self.path = path
        self.device = device
        self.threshold = threshold
        self.client = client
        self.store = None
        self.mode = "off"
        self.conversation_id = uuid.uuid4().hex
        self.sources = []
        self.turn = 0
        self.status = {}
        self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Choose off, ephemeral or explicitly persistent storage."""
        if mode not in ("off", "ephemeral", "persistent"):
            raise ValueError("Memory mode must be off, ephemeral or persistent.")
        if mode == self.mode:
            return
        if self.store:
            self.store.close()
        self.store = None
        self.status = {}
        self.sources = []
        if self.client and mode == "off":
            self.client.close()
            self.client = None
        if mode != "off":
            try:
                self.store = MemoryStore(self.path if mode == "persistent" else None)
            except Exception as error:
                self.status = {"error": str(error)}
                if self.client:
                    self.client.close()
                    self.client = None
                self.mode = mode
                return
            if self.client is None:
                try:
                    self.client = EncoderClient(self.device)
                except (OSError, RuntimeError) as error:
                    self.status = {"error": str(error)}
        self.mode = mode
        self.sources = []

    def new_conversation(self) -> str:
        """Start a fresh scope while preserving explicitly persistent records."""
        self.conversation_id = uuid.uuid4().hex
        self.turn = 0
        self.sources = []
        self.status = {}
        return self.conversation_id

    def retrieve(self, text: str, exclude: set[tuple[str, int, str]] | None = None) -> list[MemoryHit]:
        """Retrieve scoped hits, degrading to no memory on any encoder failure."""
        self.status = {"mode": self.mode, "conversation_id": self.conversation_id,
                       "latency_ms": 0.0, "references": [], "memory_tokens": 0}
        if self.mode == "off":
            return []
        started = time.perf_counter()
        try:
            if self.store is None:
                raise RuntimeError("Memory storage unavailable.")
            if self.client is None or self.client.metadata is None:
                raise RuntimeError("Memory encoder unavailable or warming up.")
            identity = self.store.ensure_index(self.client.metadata, self.client.encode, self.user_id,
                                               [self.conversation_id, *self.sources])
            vector = self.client.encode([text], "SearchQuery")
            hits = self.store.search(self.user_id, [self.conversation_id, *self.sources], vector,
                                     identity, threshold=self.threshold, exclude=exclude)
            self.status["references"] = [asdict(hit) for hit in hits]
            self.status["encoder_resources"] = getattr(self.client, "resources", {})
            self.status["embedding_cache_hits"] = getattr(self.client, "cache_hits", 0)
            return hits
        except Exception as error:
            self.status["error"] = str(error)
            return []
        finally:
            self.status["latency_ms"] = (time.perf_counter() - started) * 1000

    def remember(self, user: str, assistant: str) -> None:
        """Store completed raw turns; embed lazily on the next retrieval."""
        if self.mode != "off" and self.store is not None:
            try:
                self.store.add_turn(self.user_id, self.conversation_id, self.turn,
                                    [("user", user), ("assistant", assistant)])
            except Exception as error:
                self.status["error"] = str(error)
        self.turn += 1

    def command(self, operation: str, value: str | list[str] | None = None) -> dict:
        """Expose explicit lifecycle operations to terminal and JSON clients."""
        if operation in ("off", "ephemeral", "persistent", "on"):
            self.set_mode("ephemeral" if operation == "on" else operation)
        elif operation == "clear":
            if self.client and hasattr(self.client, "cache"):
                self.client.cache.clear()
            self.status = {}
            if self.store:
                self.store.delete_conversation(self.user_id, self.conversation_id)
        elif operation == "delete":
            if not isinstance(value, str) or not value:
                raise ValueError("Delete requires a conversation ID.")
            if self.client and hasattr(self.client, "cache"):
                self.client.cache.clear()
            self.status = {}
            if self.store:
                self.store.delete_conversation(self.user_id, value)
            self.sources = [source for source in self.sources if source != value]
        elif operation == "sources":
            if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError("Sources must be a list of conversation IDs.")
            allowed = self.store.conversations(self.user_id) if self.store else []
            if any(source not in allowed for source in value):
                raise ValueError("Source conversation is unavailable for this user.")
            self.sources = list(dict.fromkeys(value))
        elif operation == "rebuild":
            if not self.store or not self.client or not self.client.metadata:
                raise RuntimeError("Memory encoder is unavailable.")
            self.store.ensure_index(self.client.metadata, self.client.encode, self.user_id,
                                               [self.conversation_id, *self.sources])
        elif operation not in ("stats", "refs", "list"):
            raise ValueError(f"Unknown memory operation: {operation}.")
        return {**self.status, "encoder_ready": bool(self.client and self.client.metadata is not None
                                                     and not getattr(self.client, "failed", False)),
                "mode": self.mode, "conversation_id": self.conversation_id,
                "sources": self.sources,
                "conversations": self.store.conversations(self.user_id) if self.store else []}

    def close(self) -> None:
        """Release this session's worker and database."""
        if self.client:
            self.client.close()
        if self.store:
            self.store.close()
