"""Optional memory construction with an allocation-free disabled backend."""
from pathlib import Path
import uuid

class DisabledMemory:
    """Maintain chat scope/status without storage or an encoder process."""
    mode = "off"
    store = None
    client = None
    def __init__(self, *, user_id="local", path=Path("data/memory/codexa.sqlite3"), device="cpu", threshold=0.68):
        if not user_id or not -1 <= threshold <= 1:
            raise ValueError("Invalid user identity or retrieval threshold.")
        self.settings = dict(user_id=user_id, path=path, device=device, threshold=threshold)
        self.user_id = user_id
        self.path = path
        self.device = device
        self.threshold = threshold
        self.status = {}
        self.new_conversation()
    def new_conversation(self):
        self.conversation_id = uuid.uuid4().hex
        self.turn = 0
        self.sources = []
        self.status = {}
        return self.conversation_id
    def retrieve(self, text, exclude=None):
        self.status = dict(mode="off", conversation_id=self.conversation_id, latency_ms=0.0, references=[], memory_tokens=0)
        return []
    def remember(self, user, assistant):
        self.turn += 1
    def command(self, operation, value=None):
        if operation == "clear":
            self.status = {}
        elif operation == "delete":
            if not isinstance(value, str) or not value:
                raise ValueError("Delete requires a conversation ID.")
            self.status = {}
        elif operation == "sources":
            if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
                raise ValueError("Sources must be a list of conversation IDs.")
            if value:
                raise ValueError("Source conversation is unavailable for this user.")
        elif operation == "rebuild":
            raise RuntimeError("Memory encoder is unavailable.")
        elif operation not in ("off", "stats", "refs", "list"):
            raise ValueError(f"Unknown memory operation: {operation}.")
        return dict(self.status, mode="off", conversation_id=self.conversation_id, sources=[], conversations=[], encoder_ready=False)
    def close(self):
        pass


def create_memory(*, mode="off", **settings):
    """Import optional persistence only when explicitly enabled."""
    if mode == "off":
        return DisabledMemory(**settings)
    from llm_memory.service import ConversationMemory
    return ConversationMemory(mode=mode, **settings)
