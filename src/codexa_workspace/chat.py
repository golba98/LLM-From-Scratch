"""Chat with the catalog model: fill in --checkpoint/--tokenizer from configs/artifact-catalog.json,
then hand over to the native chat CLI (``python run.py module codexa_workspace.chat``)."""

import json
import sys

from llm_inference.cli.chat_native import main as chat_main

from codexa_workspace.assets import workspace_root, asset_root


def main():
    root = workspace_root()
    catalog = json.loads((root / 'configs/artifact-catalog.json').read_text())
    assets = asset_root(root)
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
    raise SystemExit(main())
