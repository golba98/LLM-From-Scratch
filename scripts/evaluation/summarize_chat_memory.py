"""Aggregate saved outputs without treating substring matches as intelligence."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

from collections import Counter
import argparse
import json
from pathlib import Path
import re
import statistics


def summarize(path: Path) -> dict:
    """Return literal-format and timing diagnostics, never human quality scores."""
    report = json.loads(path.read_text())
    groups = {}
    for enabled in (False, True):
        rows = [r for r in report['outputs'] if r['memory_enabled'] == enabled]
        if not rows:
            continue
        expected = [r for r in rows if r.get('expected')]
        exact = [r for r in expected if re.search(r'(?<!\w)' + re.escape(r['expected']) + r'(?!\w)', r['response'], re.I)]
        groups['memory_on' if enabled else 'memory_off'] = dict(
            count=len(rows), latency_seconds_median=statistics.median(r['seconds'] for r in rows),
            repeated_trigram_fraction_mean=statistics.mean(r['repeated_trigram_fraction'] for r in rows),
            termination_counts=dict(Counter(r['termination'] for r in rows)),
            END_count=sum(r.get('ended_with_END', False) for r in rows),
            END_metadata_available=all('ended_with_END' in r for r in rows),
            expected_whole_value_present=dict(matches=len(exact), queries=len(expected)),
            self_attributed_name_count=sum(bool(re.search(r'\bmy name is\s+' + re.escape(r['expected']) + r'\b', r['response'], re.I))
                                          for r in exact if r['category'] == 'long_distance'),
            exact_READY=sum(r['response'].strip() == 'READY' for r in rows if r['case'] == 6),
            retrieved_context_queries=sum(bool(r['memory'].get('references')) for r in rows),
            memory_tokens_max=max(r['memory'].get('memory_tokens', 0) for r in rows),
            generation_peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in rows),
            gpu_total_used_mib_max=max((r.get('gpu_total_used_mib', 0) for r in rows)),
        )
    return dict(source=str(path), checkpoint=report['checkpoint'], conditions=groups,
                caveat='Literal values, formatting, repetition and termination are automated diagnostics, not semantic accuracy or human ratings.')


def main() -> None:
    """Write an immutable comparison summary from existing evaluation reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [summarize(path) for path in args.report]
    with args.output.open('x') as handle:
        json.dump(results, handle, indent=2)
    print(json.dumps(results))


if __name__ == '__main__':
    main()
