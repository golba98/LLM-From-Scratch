"""Small decoding ablation; suppression of loops is not knowledge improvement."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_inference.native_chat import NativeChatEngine
from scripts.evaluation.evaluate_chat_memory import cases


def main() -> None:
    """Compare eight difficult prompts under fixed seeds and three decode policies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    engine = NativeChatEngine(checkpoint=args.checkpoint, tokenizer_path=args.tokenizer, device='cuda', maximum_new_tokens=64)
    original = engine.generation_config
    rows = []
    try:
        for label, config in [('sample', original), ('greedy', replace(original, do_sample=False)),
                              ('loop_controls', replace(original, repetition_penalty=1.15, no_repeat_ngram_size=3))]:
            engine.generation_config = config
            for i in (0, 1, 4, 5, 6, 9, 10, 11):
                engine.reset()
                prompt = cases()[i]['prompt']
                started = time.perf_counter()
                response, sequence = engine.reply(prompt)
                words = response.lower().split()
                fraction = 1 - len(set(zip(words, words[1:], words[2:]))) / max(len(words) - 2, 1) if len(words) > 2 else 0
                rows.append(dict(policy=label, settings=asdict(config), case=i, prompt=prompt,
                                 response=response, repeated_trigram_fraction=fraction,
                                 termination=sequence.termination_cause, seconds=time.perf_counter() - started))
        with args.output.open('x') as handle:
            json.dump(rows, handle, indent=2)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
