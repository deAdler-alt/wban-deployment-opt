import math
from dataclasses import dataclass

@dataclass(frozen=True)
class EnergyParams:
    E_elec: float
    E_fs: float
    E_mp: float
    E_agg: float
    packet_bits: int
    beta_agg: float  # Współczynnik kompresji
    
    @property
    def d0(self) -> float:
        return math.sqrt(self.E_fs / self.E_mp)

def calc_E_tx(k: int, d: float, p: EnergyParams) -> float:
    """Energia nadawania k bitów na odległość d."""
    if d < p.d0:
        return k * p.E_elec + k * p.E_fs * (d**2)
    else:
        return k * p.E_elec + k * p.E_mp * (d**4)

def calc_E_rx(k: int, p: EnergyParams) -> float:
    """Energia odbioru k bitów."""
    return k * p.E_elec

def calc_E_da(k: int, p: EnergyParams) -> float:
    """Energia agregacji danych."""
    return k * p.E_agg