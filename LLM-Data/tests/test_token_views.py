from pathlib import Path

import numpy as np
import pytest
import torch

from llm_data.token_data import MemmapTokenDataset, create_token_dataloader
from llm_data.token_views import IndexedTokenWindowDataset, overlapping_window_bounds


def source(tmp_path: Path, values=None):
    path = tmp_path / "tokens.bin"
    np.asarray(range(33) if values is None else values, dtype="uint16").tofile(path)
    return MemmapTokenDataset(path, dtype="uint16", context_length=4)


def test_exact_shift_and_no_copy(tmp_path):
    base = source(tmp_path)
    view = IndexedTokenWindowDataset(base, np.array([1, 4, 7]), vocab_size=64)
    x, y = view[1]
    assert x.tolist() == [16, 17, 18, 19]
    assert y.tolist() == [17, 18, 19, 20]
    assert torch.equal(view[-1][0], base[7][0])
    assert view.source is base
    with pytest.raises(IndexError):
        view[3]


@pytest.mark.parametrize("ids", [[], [0, 0], [2, 1], [-1], [8], [[0]], [0.5]])
def test_bad_selection(tmp_path, ids):
    with pytest.raises(ValueError):
        IndexedTokenWindowDataset(source(tmp_path), np.asarray(ids), vocab_size=64)


def test_target_boundary_is_also_excluded():
    assert overlapping_window_bounds(4, 5, 4, 8) == (0, 2)
    assert overlapping_window_bounds(5, 8, 4, 8) == (1, 2)
    assert overlapping_window_bounds(8, 9, 4, 8) == (1, 3)
    assert overlapping_window_bounds(32, 33, 4, 8) == (7, 8)


def test_bounds_match_exhaustive_intersections():
    for start in range(33):
        for end in range(start + 1, 34):
            first, stop = overlapping_window_bounds(start, end, 4, 8)
            expected = [i for i in range(8) if i * 4 < end and i * 4 + 5 > start]
            assert list(range(first, stop)) == expected


def test_vocab_check_includes_target(tmp_path):
    base = source(tmp_path, [0, 1, 2, 3, 100])
    view = IndexedTokenWindowDataset(base, np.array([0]), vocab_size=64)
    with pytest.raises(ValueError, match="vocabulary"):
        view[0]


def test_sampler_reconstruction_with_existing_loader(tmp_path):
    view = IndexedTokenWindowDataset(source(tmp_path), np.array([0, 2, 3, 7]), vocab_size=64)
    def order():
        loader = create_token_dataloader(view, batch_size=1, shuffle=True, seed=42)
        return [x.tolist() for x, _ in loader]
    uninterrupted = order()
    restored = order()[2:]
    assert restored == uninterrupted[2:]
