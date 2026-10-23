"""Read-only indexed views of existing packed next-token windows."""

import numpy as np
import torch
from torch.utils.data import Dataset

from llm_data.token_data import MemmapTokenDataset


def overlapping_window_bounds(start: int, end: int, context: int,
                              window_count: int) -> tuple[int, int]:
    """Return windows whose input OR shifted target intersects [start, end)."""
    if start < 0 or end <= start or context <= 0 or window_count < 0:
        raise ValueError("Invalid document interval or packed-window geometry.")
    first = max(0, (start - 1) // context)
    stop = min(window_count, (end + context - 1) // context)
    return min(first, window_count), max(min(first, window_count), stop)


class IndexedTokenWindowDataset(Dataset):
    """Select sorted unique packed windows without copying or retokenizing data.

    Selection arrays and the source file must remain immutable for the lifetime
    of a view. Selection fingerprints belong in the caller's resume identity.
    Vocabulary checks are bounded to each fetched input/target pair.
    """

    def __init__(self, source: MemmapTokenDataset, window_ids: np.ndarray,
                 *, vocab_size: int) -> None:
        ids = np.asarray(window_ids)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or not len(ids):
            raise ValueError("Window IDs must be a nonempty integer vector.")
        if int(ids.min()) < 0 or int(ids.max()) >= len(source):
            raise ValueError("Window selection escapes the source dataset.")
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("Window selection must be sorted and unique.")
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("Vocabulary size must be positive.")
        self.source = source
        self.window_ids = ids
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.window_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(index, int):
            raise TypeError("Dataset index must be an integer.")
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("Dataset index is outside the selected view.")
        inputs, labels = self.source[int(self.window_ids[index])]
        if int(inputs.max()) >= self.vocab_size or int(labels.max()) >= self.vocab_size:
            raise ValueError("Selected window contains an out-of-vocabulary token.")
        return inputs, labels
