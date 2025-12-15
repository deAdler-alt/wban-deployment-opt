import numpy as np
from .geometry import pairwise_dist

def penalty_range_sn_ch(sn_xy: np.ndarray, ch_xy: np.ndarray, 
                        assignment: np.ndarray, d_max: float, weight: float) -> float:
    """
    Liczy karę za przekroczenie zasięgu d_max dla każdego sensora.
    Penalty = sum(max(0, dist - d_max)) * weight
    """
    # Wyciągnij współrzędne przypisanych CH dla każdego sensora
    assigned_ch_xy = ch_xy[assignment]
    
    # Odległości (N,)
    # sqrt((x1-x2)^2 + ...)
    diff = sn_xy - assigned_ch_xy
    dists = np.linalg.norm(diff, axis=1)
    
    # Naruszenia
    violations = np.maximum(0.0, dists - d_max)
    
    return np.sum(violations) * weight