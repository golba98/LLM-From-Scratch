#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python "$repo/run.py" module workflows.run_recipe run_900m_sft_balanced "$@"
