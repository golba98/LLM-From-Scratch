# Codexa 934M two-hour session revision

Status: BLOCKED — configured, not launched.

The user replaced the ten-hour proposal with a two-hour session on 10 October 2026, intending further sessions tomorrow. `configs/chat-2h-v1.yaml` defines a separate run identity and 7,200-second wall-clock budget. The launch supervisor now uses the sealed specification budget rather than a hard-coded ten hours; preparation projections use the configured budget. Existing readiness gates remain enforced. No historical manifest or checkpoint was overwritten.

The prior readiness report remains the evidence record. Initialization quality, behavioral pilot improvement, held-out exposure provenance, and strict memory headroom remain unresolved. Reducing duration does not resolve these gates. Current observed free GPU memory is 13,156 MiB of 16,376 MiB; this snapshot is not a workload headroom test.

Tomorrow's additional time requires a new explicitly authorized session budget with a full recovery checkpoint, counters and sampler continuation. Ordinary resume continues to count downtime against the original deadline; it cannot silently grant another two hours. No accepted recoverable production session exists yet.

Configuration check: `.venv/bin/python scripts/chat_training.py preflight --config configs/chat-2h-v1.yaml --run codexa-900m-chat-2h-v1`. Production remains gated and has not started. The new identity has no sealed readiness manifest yet. Do not run full preparation automatically: it would repeat expensive diagnostics.
