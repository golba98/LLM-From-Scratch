"""Export a separately named native model and verify source/export equivalence."""
from llm_inference.cli.paths import asset_path, generated_path

import argparse
import gc
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from llm_architecture.checkpointing import file_sha256, verify_checkpoint_checksum, read_native_checkpoint
from llm_inference.native_chat import NativeChatEngine


def export_native(checkpoint: Path, tokenizer: Path, output: Path, *, smoke: bool = True) -> dict:
    """Preserve source checkpoints and create a weights-only native inference export."""
    if output.resolve().is_relative_to(checkpoint.parent.resolve()):
        raise ValueError('Export must be separate from the training directory.')
    source_hash = verify_checkpoint_checksum(checkpoint)
    source = read_native_checkpoint(checkpoint, map_location='cpu')
    tokenizer_hash = file_sha256(tokenizer)
    if source.get('tokenizer_sha256') != tokenizer_hash:
        raise ValueError('Export requires a matching tokenizer fingerprint.')
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'model.pt'
    exported = dict(model_state_dict=source['model_state_dict'], config=source['config'],
                    tokenizer_sha256=tokenizer_hash, tokenizer_reference='tokenizer.json')
    torch.save(exported, target)
    target.with_suffix('.pt.sha256').write_text(file_sha256(target) + '  model.pt\n')
    shutil.copyfile(tokenizer, output / 'tokenizer.json')
    report = dict(source=str(checkpoint), source_sha256=source_hash, export=str(target),
                  tokenizer_sha256=tokenizer_hash, status='experimental-not-promoted', exported_utc=time.time(),
                  model_config=source['config']['model'])
    del source, exported
    if smoke:
        sequences = []
        for model_path, tokenizer_path in ((checkpoint, tokenizer), (target, output / 'tokenizer.json')):
            engine = NativeChatEngine(checkpoint=model_path, tokenizer_path=tokenizer_path,
                                      device='cuda' if torch.cuda.is_available() else 'cpu', maximum_new_tokens=16)
            response, sequence = engine.reply('Hello!')
            sequences.append(list(sequence.visible_token_ids))
            report.setdefault('smoke_responses', []).append(response)
            engine.close()
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        report['identical_smoke_token_ids'] = sequences[0] == sequences[1]
        if sequences[0] != sequences[1]:
            raise RuntimeError('Native/exported token-level smoke equivalence failed.')
    with (output / 'export_report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    return report


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=asset_path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_native(args.checkpoint, args.tokenizer, args.output)))


if __name__ == '__main__':
    main()
