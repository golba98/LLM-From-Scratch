"""Offline persistence and worker contracts without a generative dependency."""

from dataclasses import asdict
import sys

import numpy as np
import pytest

from llm_memory.store import MemoryStore
from llm_memory.service import ConversationMemory


def vector(texts, task):
    result = np.zeros((len(texts), 768), dtype=np.float32)
    result[:, 0] = 1
    return result


def test_scoped_persistence_and_deletion(tmp_path):
    path = tmp_path / "memory.sqlite3"
    store = MemoryStore(path)
    store.add_turn("alice", "one", 0, [("user", "cat"), ("assistant", "dog")])
    store.add_turn("bob", "one", 0, [("user", "private"), ("assistant", "secret")])
    identity = store.ensure_index({"revision": "fixture"}, vector, "alice", ["one"])
    hits = store.search("alice", ["one"], vector(["cat"], "SearchQuery"), identity)
    assert len(hits) == 2
    assert {h.content for h in hits} == {"cat", "dog"}
    assert not store.search("bob", ["one"], vector(["cat"], "SearchQuery"), identity)
    before = [asdict(h) for h in hits]
    store.close()
    reopened = MemoryStore(path)
    assert [asdict(h) for h in reopened.search("alice", ["one"], vector(["cat"], "SearchQuery"), identity)] == before
    assert reopened.delete_conversation("alice", "one") == 2
    assert reopened.conversations("bob") == ["one"]
    assert not reopened.search("alice", ["one"], vector(["cat"], "SearchQuery"), identity)
    reopened.close()


def test_completed_turn_is_immutable(tmp_path):
    store = MemoryStore(tmp_path / "memory.sqlite3")
    turn = [("user", "cat"), ("assistant", "dog")]
    store.add_turn("alice", "one", 0, turn)
    store.add_turn("alice", "one", 0, turn)
    with pytest.raises(ValueError, match="overwritten"):
        store.add_turn("alice", "one", 0, [("user", "cat"), ("assistant", "changed")])
    store.close()


def test_worker_failure_preserves_original_turns():
    class Unavailable:
        metadata = None
        def close(self):
            pass
    memory = ConversationMemory(mode="ephemeral", client=Unavailable())
    memory.remember("cat", "dog")
    assert memory.retrieve("cat") == []
    assert "error" in memory.status
    assert memory.store.conversations("local") == [memory.conversation_id]
    memory.close()


def test_import_has_no_model_dependency():
    assert "llm_architecture" not in sys.modules
    assert "llm_specialist" not in sys.modules
