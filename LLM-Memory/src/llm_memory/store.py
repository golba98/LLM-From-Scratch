"""User-scoped SQLite records and exact cosine retrieval."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class MemoryHit:
    """A ranked excerpt retaining its source and chronological position."""

    id: str
    conversation_id: str
    turn: int
    role: str
    content: str
    score: float
    created_at: float = 0.0


def unit_vectors(values, count: int) -> np.ndarray:
    """Validate finite, normalized FP32 vectors before indexing or scoring."""
    array = np.asarray(values)
    if array.dtype == np.float16:
        raise ValueError("FP16 embeddings are forbidden.")
    array = array.astype(np.float32)
    if array.shape != (count, 768) or not np.isfinite(array).all():
        raise ValueError("Expected finite [count, 768] embeddings.")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1, atol=0.002):
        raise ValueError("Embeddings must be normalized.")
    return array / norms[:, None]


class MemoryStore:
    """Persist original records independently of versioned embedding indexes."""

    def __init__(self, path: Path | None = None) -> None:
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
        existed = path is not None and path.exists()
        self.connection = sqlite3.connect(":memory:" if path is None else str(path))
        if existed:
            expected = {
                "records": {"id", "user_id", "conversation", "turn", "role", "content", "created", "digest"},
                "turns": {"user_id", "conversation", "turn", "messages"},
                "indexes": {"identity", "metadata"},
                "vectors": {"record_id", "identity", "value"},
            }
            try:
                for table, columns in expected.items():
                    found = {row[1] for row in self.connection.execute(f"PRAGMA table_info({table})")}
                    if found != columns:
                        raise ValueError("Existing file is not a compatible conversation memory database.")
            except Exception:
                self.connection.close()
                raise
        elif path is not None:
            path.chmod(0o600)
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.execute("PRAGMA secure_delete=ON")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS turns (user_id TEXT, conversation TEXT, turn INTEGER,
                messages TEXT NOT NULL, PRIMARY KEY(user_id, conversation, turn));
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY, user_id TEXT NOT NULL, conversation TEXT NOT NULL,
                turn INTEGER NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
                created REAL NOT NULL, digest TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS scope ON records(user_id, conversation, turn);
            CREATE TABLE IF NOT EXISTS indexes (identity TEXT PRIMARY KEY, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS vectors (
                record_id TEXT REFERENCES records(id) ON DELETE CASCADE,
                identity TEXT REFERENCES indexes(identity), value BLOB NOT NULL,
                PRIMARY KEY(record_id, identity));
        """)
        self.connection.commit()

    def add_turn(self, user_id: str, conversation: str, turn: int,
                 messages: list[tuple[str, str]]) -> None:
        """Save an idempotent complete turn without overwriting original content."""
        if not user_id or not conversation or turn < 0:
            raise ValueError("User, conversation and nonnegative turn are required.")
        if [role for role, _ in messages] != ["user", "assistant"]:
            raise ValueError("Memory requires a complete user/assistant turn.")
        with self.connection:
            raw = json.dumps(messages, ensure_ascii=False)
            existing = self.connection.execute("SELECT messages FROM turns WHERE user_id=? AND conversation=? AND turn=?",
                                               (user_id, conversation, turn)).fetchone()
            if existing and existing[0] != raw:
                raise ValueError("A completed memory turn cannot be overwritten.")
            self.connection.execute("INSERT OR IGNORE INTO turns VALUES (?,?,?,?)", (user_id, conversation, turn, raw))
            for role, content in messages:
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Memory content must be nonempty.")
                # A 480-byte cap also bounds byte-fallback token counts; retain exact text.
                chunks = []
                buffer = ""
                for character in content:
                    if len((buffer + character).encode("utf-8")) > 480:
                        chunks.append(buffer)
                        buffer = ""
                    buffer += character
                if buffer:
                    chunks.append(buffer)
                for chunk, text in enumerate(chunks):
                    record_id = hashlib.sha256(json.dumps([user_id, conversation, turn, role, chunk]).encode()).hexdigest()
                    digest = hashlib.sha256(text.encode()).hexdigest()
                    existing = self.connection.execute("SELECT digest FROM records WHERE id=?", (record_id,)).fetchone()
                    if existing and existing[0] != digest:
                        raise ValueError("A completed memory turn cannot be overwritten.")
                    self.connection.execute("INSERT OR IGNORE INTO records VALUES (?,?,?,?,?,?,?,?)",
                                            (record_id, user_id, conversation, turn, role, text, time.time(), digest))

    def ensure_index(self, metadata: dict, encode: Callable, user_id: str, conversations: list[str] | None = None) -> str:
        """Build a new identity transactionally; retain all old index revisions."""
        serialized = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        identity = hashlib.sha256(serialized.encode()).hexdigest()
        scope = ""
        parameters = [user_id, identity]
        if conversations is not None:
            scope = " AND conversation IN (" + ",".join("?" for _ in conversations) + ")"
            parameters.extend(conversations)
        pending = self.connection.execute("""SELECT id, role, content FROM records
            WHERE user_id=? AND id NOT IN (SELECT record_id FROM vectors WHERE identity=?)"""
            + scope + " ORDER BY id", parameters).fetchall()
        # Compute all missing batches before changing the active identity.
        batches = []
        for start in range(0, len(pending), 16):
            rows = pending[start:start + 16]
            vectors = unit_vectors(encode([f"{r[1]}: {r[2]}" for r in rows], "Document"), len(rows))
            batches.extend((r[0], identity, v.tobytes()) for r, v in zip(rows, vectors))
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO indexes VALUES (?,?)", (identity, serialized))
            self.connection.executemany("INSERT OR IGNORE INTO vectors VALUES (?,?,?)", batches)
        return identity

    def search(self, user_id: str, conversations: list[str], query, identity: str,
               *, limit: int = 3, threshold: float = 0.6,
               exclude: set[tuple[str, int, str]] | None = None) -> list[MemoryHit]:
        """Filter scopes before scoring; break ties by stable source identity."""
        if not conversations or limit < 1:
            return []
        if not -1 <= threshold <= 1:
            raise ValueError("Similarity threshold must be in [-1, 1].")
        vector = unit_vectors(query, 1)[0]
        placeholders = ",".join("?" for _ in conversations)
        rows = self.connection.execute(f"""SELECT r.id,r.conversation,r.turn,r.role,r.content,r.digest,v.value,r.created
            FROM records r JOIN vectors v ON r.id=v.record_id
            WHERE r.user_id=? AND r.conversation IN ({placeholders}) AND v.identity=?""",
            [user_id, *conversations, identity]).fetchall()
        candidates = []
        for row in rows:
            if exclude and (row[1], row[2], row[3]) in exclude:
                continue
            value = unit_vectors(np.frombuffer(row[6], dtype=np.float32)[None, :], 1)[0]
            score = float(value @ vector)
            if score >= threshold:
                candidates.append((score, row))
        candidates.sort(key=lambda item: (-item[0], item[1][0]))
        seen = set()
        hits = []
        for score, row in candidates:
            key = (row[1], row[2], row[3])
            if key in seen or row[5] in seen:
                continue
            seen.update((key, row[5]))
            hits.append(MemoryHit(row[0], row[1], row[2], row[3], row[4], score, row[7]))
            if len(hits) == limit:
                break
        return hits

    def delete_conversation(self, user_id: str, conversation: str) -> int:
        """Delete only the requested user's conversation and cascading vectors."""
        with self.connection:
            self.connection.execute("DELETE FROM turns WHERE user_id=? AND conversation=?", (user_id, conversation))
            return self.connection.execute("DELETE FROM records WHERE user_id=? AND conversation=?",
                                           (user_id, conversation)).rowcount

    def conversations(self, user_id: str) -> list[str]:
        """List this user's saved conversation identities."""
        return [r[0] for r in self.connection.execute(
            "SELECT DISTINCT conversation FROM records WHERE user_id=? ORDER BY conversation", (user_id,))]

    def close(self) -> None:
        """Close this owned SQLite connection."""
        self.connection.close()
