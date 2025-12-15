from __future__ import annotations

import numpy as np


def penalty_range_sn_ch(
    sn_xy: np.ndarray,
    ch_xy: np.ndarray,
    assign: np.ndarray,
    dmax: float,
    lam: float,
) -> float:
    """
    Miękka kara: jeśli sensor -> przypisany CH ma dystans > dmax,
    dodajemy karę lam * (d - dmax)^2.

    Zwraca skalar.
    """
    sn_xy = np.asarray(sn_xy, dtype=float)
    ch_xy = np.asarray(ch_xy, dtype=float)
    assign = np.asarray(assign, dtype=int)

    if dmax <= 0:
        raise ValueError("dmax musi być dodatnie")
    if lam < 0:
        raise ValueError("lam nie może być ujemne")

    # dystans SN do przypisanego CH
    chosen = ch_xy[assign]                      # (N x 2)
    d = np.linalg.norm(sn_xy - chosen, axis=1)  # (N,)

    excess = np.maximum(0.0, d - float(dmax))
    return float(lam * np.sum(excess ** 2))
