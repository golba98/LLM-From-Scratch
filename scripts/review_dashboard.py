"""Publish a public-safe human review dashboard from an existing review package."""

import argparse
import json
from pathlib import Path
import shlex
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from codexa_workspace.review_dashboard import publish_public_review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('publish',))
    parser.add_argument('--source', type=Path, required=True, help='Existing private review directory with package.json')
    parser.add_argument('--output', type=Path, required=True, help='Fresh public directory (receives only index.html)')
    parser.add_argument('--mapping', type=Path, help='Private mapping, read only to verify nothing leaks')
    parser.add_argument('--title', default='Codexa — Model Evaluation')
    args = parser.parse_args()
    result = publish_public_review(args.source, args.output, mapping=args.mapping, title=args.title)
    result['serve'] = f'npx live-server --host=127.0.0.1 --port=8080 --no-browser {shlex.quote(result["directory"])}'
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
