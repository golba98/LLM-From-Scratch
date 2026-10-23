"""Exercise the real subprocess transport with an offline fixture worker."""

import shlex
import sys
import time

from llm_memory.client import EncoderClient


def test_configured_worker_cache_and_shutdown(tmp_path, monkeypatch):
    worker = tmp_path / "worker fixture.py"
    worker.write_text('''import json, sys
print(json.dumps({"ready": {"dimension": 768, "revision": "fixture"}}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    if request.get("shutdown"):
        break
    print(json.dumps({"vectors": [[1.0] + [0.0] * 767 for _ in request["texts"]]}), flush=True)
''')
    monkeypatch.setenv("LLM_MEMORY_WORKER_COMMAND", shlex.join([sys.executable, str(worker)]))
    client = EncoderClient(timeout=2)
    try:
        deadline = time.monotonic() + 3
        while client.metadata is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert client.metadata["revision"] == "fixture"
        first = client.encode(["cat"], "SearchQuery")
        second = client.encode(["cat"], "SearchQuery")
        first[0][0] = 0
        assert second[0][0] == 1 and client.cache_hits == 1
    finally:
        client.close()
    assert client.process.poll() == 0
