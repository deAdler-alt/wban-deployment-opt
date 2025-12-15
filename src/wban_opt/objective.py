from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .config import Point
from .geometry import points_xy
from .repair import decode_to_indices, repair_unique
from .assignment import assign_sensors_to_ch
from .penalties import penalty_range_sn_ch
from .energy_model import EnergyParams, E_tx, E_rx, E_da


@dataclass(frozen=True)
class ObjectiveContext:
    # P: lista punktów montażu (M punktów)
    points: List[Point]

    # N: liczba sensorów
    N: int

    # K: liczba koncentratorów / CH
    K: int

    # pozycja bramki (GW)
    gw_xy: Tuple[float, float]

    # parametry energetyczne
    energy: EnergyParams

    # constraint zasięgu SN->CH
    d_max_sn_ch: float

    # współczynnik kary (miękki constraint)
    penalty_range: float


def energy_and_penalty_from_x(x: np.ndarray, ctx: ObjectiveContext) -> tuple[float, float]:
    """
    Zwraca (E_total, Penalty) dla wektora x długości D=N+K.

    - x ∈ [0,1]^(N+K) (Mealpy)
    - dekodowanie -> indeksy punktów z P
    - naprawa unikalności (brak kolizji indeksów)
    - przypisanie sensorów do najbliższego CH
    - energia: SN TX do CH, CH RX+DA+TX do GW
    - kara: przekroczenia d_max dla SN->CH
    """
    x = np.asarray(x, dtype=float)
    M = len(ctx.points)
    D = int(ctx.N) + int(ctx.K)

    if x.shape[0] != D:
        raise ValueError(f"Zły wymiar x: {x.shape[0]} != D={D}")

    # 1) mapowanie x -> indeksy 0..M-1
    idx = decode_to_indices(x, M)

    # 2) naprawa unikalności (SN+CH nie mogą wskazywać tego samego punktu)
    idx = repair_unique(idx, M)

    # 3) indeksy -> współrzędne
    p_xy = points_xy(ctx.points)               # (M,2)
    sn_idx = idx[: ctx.N]
    ch_idx = idx[ctx.N :]

    sn_xy = p_xy[sn_idx]                       # (N,2)
    ch_xy = p_xy[ch_idx]                       # (K,2)
    gw_xy = np.asarray(ctx.gw_xy, dtype=float) # (2,)

    # 4) przypisanie sensorów do najbliższego CH
    assign = assign_sensors_to_ch(sn_xy, ch_xy)  # (N,) int w [0..K-1]

    k = int(ctx.energy.packet_bits)
    beta = float(ctx.energy.beta_agg)

    # 5) energia sensorów: TX do CH
    E_sensors = 0.0
    for i in range(ctx.N):
        j = int(assign[i])
        d = float(np.linalg.norm(sn_xy[i] - ch_xy[j]))
        E_sensors += E_tx(k, d, ctx.energy)

    # 6) energia CH: RX od członków + agregacja + TX do GW
    E_ch = 0.0
    for j in range(ctx.K):
        members = np.where(assign == j)[0]
        m = int(len(members))
        if m == 0:
            continue

        E_ch += m * E_rx(k, ctx.energy)
        E_ch += m * E_da(k, ctx.energy)

        k_aggr = int(beta * m * k)
        d_gw = float(np.linalg.norm(ch_xy[j] - gw_xy))
        E_ch += E_tx(k_aggr, d_gw, ctx.energy)

    E_total = float(E_sensors + E_ch)

    # 7) kara za przekroczenie zasięgu SN->CH (miękki constraint)
    P = penalty_range_sn_ch(
        sn_xy=sn_xy,
        ch_xy=ch_xy,
        assign=assign,
        dmax=float(ctx.d_max_sn_ch),
        lam=float(ctx.penalty_range),
    )

    return E_total, float(P)


def objective_from_x(x: np.ndarray, ctx: ObjectiveContext) -> float:
    """
    Funkcja celu (fitness) dla Mealpy: minimalizujemy E_total + Penalty.
    """
    E_total, P = energy_and_penalty_from_x(x, ctx)
    return float(E_total + P)
