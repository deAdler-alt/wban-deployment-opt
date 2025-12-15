from __future__ import annotations

import numpy as np
from .geometry import pairwise_dist


def assign_sensors_to_ch(sn_xy: np.ndarray, ch_xy: np.ndarray) -> np.ndarray:
    """
    Przypisuje każdy sensor do najbliższego CH (argmin po odległości).

    sn_xy: (N x 2)
    ch_xy: (K x 2)
    zwraca: (N,) indeksy 0..K-1
    """
    sn_xy = np.asarray(sn_xy, dtype=float)
    ch_xy = np.asarray(ch_xy, dtype=float)

    if sn_xy.ndim != 2 or sn_xy.shape[1] != 2:
        raise ValueError("sn_xy musi mieć kształt (N,2)")
    if ch_xy.ndim != 2 or ch_xy.shape[1] != 2:
        raise ValueError("ch_xy musi mieć kształt (K,2)")
    if ch_xy.shape[0] == 0:
        raise ValueError("K=0 (brak CH) jest niedozwolone")

    d = pairwise_dist(sn_xy, ch_xy)          # (N x K)
    return np.argmin(d, axis=1).astype(int)  # (N,)
