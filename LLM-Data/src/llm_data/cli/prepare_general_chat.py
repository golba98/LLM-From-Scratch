"""Prepare Stage 2 conversational data from existing local source derivatives."""
from llm_data.cli.paths import asset_path, generated_path

import argparse
import json
from pathlib import Path
import sys

from llm_data.data.general_chat import prepare_general_chat
from llm_tokenizer.tokenizer import load_tokenizer


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, action='append')
    parser.add_argument('--output', type=Path, default=generated_path('data/processed/general-chat-sft-v2'))
    parser.add_argument('--tokenizer', type=Path, default=asset_path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    args = parser.parse_args()
    paths = args.input or sorted(asset_path('data/processed/chat-sft-v1').glob('*/train.jsonl')) + sorted(asset_path('data/processed/chat-sft-v1').glob('*/validation.jsonl'))
    print(json.dumps(prepare_general_chat(paths, args.output, load_tokenizer(args.tokenizer))))


if __name__ == '__main__':
    main()
