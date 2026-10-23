"""Bounded asynchronous local subprocess protocol for retrieval embeddings."""

import json
import os
import shlex
import sys
from collections import OrderedDict
import copy
import hashlib
from pathlib import Path
import queue
import subprocess
import threading


class EncoderClient:
    """Load the specialist in the background; serialize bounded requests."""

    def __init__(self, device: str = "cpu", *, python: Path | None = None,
                 timeout: float = 10.0) -> None:
        configured = os.environ.get("LLM_MEMORY_WORKER_COMMAND")
        command = shlex.split(configured) if configured else [
            str(python or os.environ.get("LLM_MEMORY_WORKER_PYTHON") or sys.executable),
            "-m", "llm_specialist.cli.memory_worker", "--device", device,
        ]
        if not command:
            raise ValueError("Memory worker command must not be empty.")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, bufsize=1)
        self.timeout = timeout
        self.responses = queue.Queue()
        self.metadata = None
        self.resources = {}
        self.failed = False
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self) -> None:
        try:
            for line in self.process.stdout:
                value = json.loads(line)
                self.resources = value.get("resources", self.resources)
                if "ready" in value:
                    self.metadata = value["ready"]
                else:
                    self.responses.put(value)
        except Exception:
            self.failed = True
        finally:
            self.failed = True
            self.responses.put({"error": "Encoder worker exited."})

    def encode(self, texts: list[str], task: str) -> list[list[float]]:
        """Return vectors or fail without blocking ordinary native generation."""
        if self.failed or self.metadata is None:
            raise RuntimeError("Memory encoder unavailable or warming up.")
        with self.lock:
            key = hashlib.sha256(json.dumps([task, texts], ensure_ascii=False).encode()).hexdigest()
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache_hits += 1
                return copy.deepcopy(self.cache[key])
            self.process.stdin.write(json.dumps({"texts": texts, "task": task}) + "\n")
            self.process.stdin.flush()
            try:
                response = self.responses.get(timeout=self.timeout)
            except queue.Empty as error:
                self.close()  # Never associate a late reply with a subsequent request.
                raise TimeoutError("Memory encoder request timed out.") from error
            if "error" in response:
                raise RuntimeError(response["error"])
            self.cache[key] = response["vectors"]
            if len(self.cache) > 128:
                self.cache.popitem(last=False)
            return copy.deepcopy(response["vectors"])

    def close(self) -> None:
        """Stop only this client's owned worker and release GPU/CPU resources."""
        self.failed = True
        self.cache.clear()
        if self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=3)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
