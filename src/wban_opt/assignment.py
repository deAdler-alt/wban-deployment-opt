import numpy as np
from .geometry import pairwise_dist

def assign_sensors_to_ch(sn_xy: np.ndarray, ch_xy: np.ndarray) -> np.ndarray:
    """
    Przypisuje każdy sensor do najbliższego CH.
    Zwraca: wektor indeksów (długość N) wskazujący index w ch_xy (0..K-1).
    """
    if ch_xy.shape[0] == 0:
        raise ValueError("Brak CH do przypisania!")
    
    dists = pairwise_dist(sn_xy, ch_xy)  # Shape (N, K)
    assignment = np.argmin(dists, axis=1)
    return assignment