from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class EnergyParams:
    """
    Parametry First Order Radio Model (wariant d^2 / d^4) + agregacja.
    Jednostki:
      - E_* w J/bit lub J/bit/m^n
      - d w metrach
      - k w bitach
    """
    E_elec: float     # J/bit
    E_fs: float       # J/bit/m^2
    E_mp: float       # J/bit/m^4
    E_agg: float      # J/bit (agregacja)
    packet_bits: int  # bity/pakiet
    beta_agg: float   # współczynnik agregacji (0..1..), 1.0 => brak kompresji
    E_init: float     # J (energia początkowa)


def d0(params: EnergyParams) -> float:
    """Odległość progowa d0 = sqrt(E_fs / E_mp)."""
    return math.sqrt(params.E_fs / params.E_mp)


def E_tx(k: int, d: float, params: EnergyParams) -> float:
    """
    Energia transmisji k bitów na odległość d.
    d<d0 => Free-space (d^2)
    d>=d0 => Multipath (d^4)
    """
    if d < 0:
        raise ValueError("Odległość d nie może być ujemna")

    if d < d0(params):
        return k * params.E_elec + k * params.E_fs * (d ** 2)
    return k * params.E_elec + k * params.E_mp * (d ** 4)


def E_rx(k: int, params: EnergyParams) -> float:
    """Energia odbioru k bitów (nie zależy od odległości)."""
    return k * params.E_elec


def E_da(k: int, params: EnergyParams) -> float:
    """Energia agregacji k bitów."""
    return k * params.E_agg
