import numpy as np
import pytest
from wban_opt.repair import decode_to_indices, repair_unique

def test_decode_to_indices():
    M = 10
    x = np.array([0.0, 0.5, 0.9999, 0.1])
    idx = decode_to_indices(x, M)
    assert np.all(idx >= 0)
    assert np.all(idx < M)
    assert idx[0] == 0
    assert idx[1] == 5
    assert idx[2] == 9

def test_repair_unique_simple():
    M = 5
    # D=3, brak kolizji
    idx = np.array([0, 1, 2])
    res = repair_unique(idx, M)
    assert np.array_equal(idx, res)

def test_repair_unique_conflict():
    M = 5
    # D=3, kolizja [0, 0, 1]
    # Dostępne: {2, 3, 4}
    idx = np.array([0, 0, 1])
    res = repair_unique(idx, M, rng=np.random.default_rng(42))
    
    assert len(np.unique(res)) == 3
    assert np.all(res < M)
    # 0 i 1 muszą zostać (ewentualnie przesunięte), jeden 0 musi zniknąć
    assert np.sum(res == 0) == 1
    assert np.sum(res == 1) == 1