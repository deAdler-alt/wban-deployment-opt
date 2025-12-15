from __future__ import annotations

import numpy as np


def decode_to_indices(x: np.ndarray, M: int) -> np.ndarray:
    """
    Mapuje x (D,) na indeksy 0..M-1.
    D = N + K
    Działa stabilnie nawet jeśli algorytm wygeneruje wartości poza [0,1].

    Reguła:
    - clip do [0, 1)
    - idx = floor(x * M)
    """
    x = np.asarray(x, dtype=float)
    if M <= 0:
        raise ValueError("M musi być dodatnie")
    # clip do [0, 1-eps], żeby floor nie dawał M
    eps = np.finfo(float).eps
    xc = np.clip(x, 0.0, 1.0 - eps)
    idx = np.floor(xc * M).astype(int)
    return idx


def repair_unique(idx: np.ndarray, M: int) -> np.ndarray:
    """
    Naprawa unikalności indeksów (SN+CH nie mogą wskazywać tego samego punktu).
    Zakłada, że D <= M.

    Strategia:
    - zachowaj pierwsze wystąpienia
    - duplikaty zamień na kolejne wolne indeksy
    """
    idx = np.asarray(idx, dtype=int).copy()
    D = idx.shape[0]
    if D > M:
        raise ValueError(f"Nie da się zapewnić unikalności: D={D} > M={M}")

    used = set()
    # wolne indeksy
    free = [i for i in range(M) if i not in set(idx)]
    free_it = iter(free)

    for i in range(D):
        v = int(idx[i])
        if v not in used:
            used.add(v)
            continue
        # duplikat -> bierzemy pierwszy wolny
        idx[i] = next(free_it)
        used.add(int(idx[i]))

    return idx
