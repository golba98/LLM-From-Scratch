"""Benchmark host-side throughput of a memory-mapped token dataset."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np



from llm_data.token_data import (
    MemmapTokenDataset,
    benchmark_token_dataloader,
    create_token_dataloader,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--dtype", choices=("uint16", "uint32"), required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    arguments = parser.parse_args()

    dataset = MemmapTokenDataset(
        arguments.token_file,
        dtype=np.dtype(arguments.dtype),
        context_length=arguments.context_length,
    )
    loader = create_token_dataloader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
    )
    result = benchmark_token_dataloader(
        loader,
        max_batches=arguments.max_batches,
    )
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
