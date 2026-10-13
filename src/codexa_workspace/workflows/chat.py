"""Resolve catalog defaults and delegate to the complete native chat CLI."""

import json
from pathlib import Path
import sys

from llm_inference.cli.chat_native import main as chat_main

from codexa_workspace.assets import workspace_root, asset_root, resolve_input
ROOT = workspace_root()


def main():
    catalog = json.loads((ROOT / 'configs/artifact-catalog.json').read_text())
    assets = asset_root(ROOT)
    checkpoint = assets / catalog['chat_baseline']['path']
    tokenizer = assets / catalog['tokenizer']['path']
    arguments = sys.argv[1:]
    if '--describe' in arguments:
        print(json.dumps(dict(checkpoint=str(checkpoint), tokenizer=str(tokenizer),
                              status='experimental comparison baseline'), indent=2))
        return
    defaults = []
    for flag, path in [('--checkpoint', checkpoint), ('--tokenizer', tokenizer)]:
        if not any(arg == flag or arg.startswith(flag + '=') for arg in arguments):
            defaults.extend([flag, str(path)])
    sys.argv = [sys.argv[0], *defaults, *arguments]
    return chat_main()


if __name__ == '__main__':
    main()
