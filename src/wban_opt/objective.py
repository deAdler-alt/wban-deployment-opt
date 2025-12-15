import numpy as np
from dataclasses import dataclass
from .geometry import points_to_numpy, pairwise_dist
from .assignment import assign_sensors_to_ch
from .energy_model import EnergyParams, calc_E_tx, calc_E_rx, calc_E_da
from .penalties import penalty_range_sn_ch
from .repair import decode_to_indices, repair_unique

@dataclass(frozen=True)
class ObjectiveContext:
    points_pool: np.ndarray  # (M, 2)
    N: int
    K: int
    gw_xy: np.ndarray        # (1, 2)
    energy_params: EnergyParams
    d_max_sn_ch: float
    penalty_weight: float
    
    def get_D(self):
        return self.N + self.K

def energy_and_penalty_from_x(x: np.ndarray, ctx: ObjectiveContext) -> tuple[float, float, bool]:
    """
    Główna logika: x -> (Energy, Penalty, Feasible)
    """
    M = ctx.points_pool.shape[0]
    
    # 1. Decode & Repair
    raw_idx = decode_to_indices(x, M)
    clean_idx = repair_unique(raw_idx, M) # Strategia losowa, ale wewnątrz funkcji celu determinizm wymagałby fixed seed
                                          # W Mealpy dla stochastyczności OK, dla evaluacji końcowej ostrożnie.
                                          # Tutaj zakładamy "miękkie" repair dla procesu ewolucji.
    
    # 2. Split SN / CH
    sn_indices = clean_idx[:ctx.N]
    ch_indices = clean_idx[ctx.N:]
    
    sn_xy = ctx.points_pool[sn_indices]
    ch_xy = ctx.points_pool[ch_indices]
    
    # 3. Assignment
    assignment = assign_sensors_to_ch(sn_xy, ch_xy)
    
    # 4. Energy Calculation
    # A. Sensors TX
    assigned_ch_xy = ch_xy[assignment]
    dists_sn_ch = np.linalg.norm(sn_xy - assigned_ch_xy, axis=1)
    
    E_sn_total = 0.0
    for d in dists_sn_ch:
        E_sn_total += calc_E_tx(ctx.energy_params.packet_bits, d, ctx.energy_params)
        
    # B. CH Nodes (RX + Agg + TX to GW)
    E_ch_total = 0.0
    # Policz ile sensorów na każdy CH
    counts = np.bincount(assignment, minlength=ctx.K)
    
    dists_ch_gw = np.linalg.norm(ch_xy - ctx.gw_xy, axis=1)
    
    for j in range(ctx.K):
        m_j = counts[j] # liczba sensorów podpiętych
        if m_j > 0:
            # RX
            E_ch_total += calc_E_rx(ctx.energy_params.packet_bits * m_j, ctx.energy_params)
            # DA
            E_ch_total += calc_E_da(ctx.energy_params.packet_bits * m_j, ctx.energy_params)
            # TX -> GW
            k_aggr = ctx.energy_params.packet_bits * m_j * ctx.energy_params.beta_agg
            E_ch_total += calc_E_tx(k_aggr, dists_ch_gw[j], ctx.energy_params)
            
    E_total = E_sn_total + E_ch_total
    
    # 5. Penalties
    P = penalty_range_sn_ch(sn_xy, ch_xy, assignment, ctx.d_max_sn_ch, ctx.penalty_weight)
    
    is_feasible = (P == 0.0)
    
    return E_total, P, is_feasible

def objective_from_x(x: np.ndarray, ctx: ObjectiveContext) -> float:
    E, P, _ = energy_and_penalty_from_x(x, ctx)
    return E + P