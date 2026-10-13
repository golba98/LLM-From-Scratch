"""Save real native outputs with equivalent decoding, separating retrieval and generation."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_bootstrap_sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))
import workspace_bootstrap

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import resource
import re
import statistics
import sys
import subprocess
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from llm_inference.native_chat import NativeChatEngine
from llm_memory.service import ConversationMemory
from llm_tokenizer.sft import ChatMessage


def cases() -> list[dict]:
    """Evaluation-only prompts; never used in SFT preparation."""
    output = []
    for category, prompts in {
        'ordinary': ['Hello! How are you?', 'Thanks for your help.', 'I feel nervous about meeting new people.', 'What can we talk about today?'],
        'instruction': ['List exactly three fruits, one per line.', 'Explain rain in exactly two short sentences.', 'Reply with only the word READY.', 'Give one practical tip for sleeping well.'],
        'knowledge': ['What is the capital of Japan?', 'How many days are in a normal year?', 'What do plants need for photosynthesis?', 'Why does ice float on water?'],
    }.items():
        for prompt in prompts:
            output.append(dict(category=category, prompt=prompt, history=[]))
    for prompt in ['What color did I choose?', 'What did I ask you to explain?', 'Can you say that more briefly?', 'What number did I mention?']:
        history = [('user', 'I chose purple. Explain how rain forms. My number is 47.'), ('assistant', 'You chose purple and mentioned 47. Rain forms when condensed water droplets fall from clouds.')]
        output.append(dict(category='followup', prompt=prompt, history=history))
    filler = [('user', 'Let us discuss daily routines. ' * 80), ('assistant', 'Daily routines can include exercise, meals, reading, and rest. ' * 50)] * 6
    for name in ['Mara', 'Leon', 'Anika', 'Sam']:
        output.append(dict(category='long_distance', prompt='What name did I tell you earlier?',
                           history=[('user', f'My name is {name}.'), ('assistant', f'Your name is {name}.'), *filler], expected=name))
    for drink in ['green tea', 'water', 'chamomile tea', 'orange juice']:
        output.append(dict(category='correction', prompt='What is my updated morning drink preference now?',
                           history=[('user', 'I prefer coffee in the morning.'), ('assistant', 'You prefer coffee.'),
                                    ('user', f'Update: I stopped drinking coffee. I now prefer {drink}.'),
                                    ('assistant', f'Your updated preference is {drink}.'), *filler], expected=drink))
    return output


def main() -> None:
    """Run the explicit command-line operation with validated inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, default=Path('checkpoints/tokenizer-base-v1/tokenizer.json'))
    parser.add_argument('--memory-device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--only-memory', action='store_true')
    parser.add_argument('--deadline-utc', type=float)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    memory = ConversationMemory(mode='ephemeral', device=args.memory_device)
    engine = NativeChatEngine(checkpoint=args.checkpoint, tokenizer_path=args.tokenizer,
                              device='cuda', maximum_new_tokens=args.max_new_tokens, memory=memory)
    load_seconds = time.perf_counter() - started
    deadline = time.monotonic() + 60
    while (memory.client and memory.client.metadata is None and not memory.client.failed
           and time.monotonic() < deadline):
        time.sleep(0.1)
    if not memory.client or memory.client.metadata is None:
        engine.close()
        raise RuntimeError('Real retrieval encoder did not initialize for evaluation.')
    rows = []
    try:
        for enabled in ([True] if args.only_memory else [False, True]):
            for i, case in enumerate(cases()[:args.limit]):
                if args.deadline_utc and time.time() >= args.deadline_utc:
                    break
                engine.reset()
                memory.conversation_id = f"benchmark-case-{i}"
                engine.messages = [ChatMessage(*item) for item in case['history']]
                memory.turn = len(case['history']) // 2
                if enabled:
                    for turn in range(memory.turn):
                        memory.store.add_turn(memory.user_id, memory.conversation_id, turn,
                                              case['history'][turn * 2:turn * 2 + 2])
                # Leave the encoder loaded in both conditions; record actual combined memory.
                memory.mode = 'ephemeral' if enabled else 'off'
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                response, generated = engine.reply(case['prompt'])
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                words = response.lower().split()
                row = dict(case=i, category=case['category'], prompt=case['prompt'], memory_enabled=enabled,
                           response=response, expected=case.get('expected'),
                           expected_literal_present=bool(re.search(r'(?<!\w)' + re.escape(case['expected']) + r'(?!\w)', response, re.I)) if case.get('expected') else None,
                           seconds=elapsed, generated_tokens=generated.generated_token_count,
                           termination=generated.termination_cause,
                           terminating_token_id=generated.terminating_token_id,
                           ended_with_END=generated.terminating_token_id == engine.end_token_id,
                           repeated_trigram_fraction=1 - len(set(zip(words, words[1:], words[2:]))) / max(len(words) - 2, 1) if len(words) > 2 else 0,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(), memory=memory.status,
                           worker_resources=memory.client.resources,
                           gpu_total_used_mib=int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()[0]))
                rows.append(row)
                with (args.output / 'outputs.jsonl').open('a') as handle:
                    handle.write(json.dumps(row, allow_nan=False) + '\n')
                print(f'case {i} memory={enabled}: {response[:120]!r}', flush=True)
        report = dict(checkpoint=str(args.checkpoint), load_seconds=load_seconds,
                      memory_device=args.memory_device, process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                      settings=asdict(engine.generation_config), outputs=rows,
                      limitations=['Synthetic scenarios; literal expected-answer checks are not semantic judgments.',
                                   'Encoder remains loaded for both conditions to keep hardware comparable.',
                                   'No human quality ratings have been collected.'])
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    finally:
        engine.close()


if __name__ == '__main__':
    main()
