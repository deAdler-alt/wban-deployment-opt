import numpy as np
from scipy.spatial.distance import cdist
from .config import Point

def points_to_numpy(points: list[Point]) -> np.ndarray:
    """Konwertuje listę Point na macierz (M, 2)."""
    coords = [[p.x, p.y] for p in points]
    return np.array(coords)

def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Oblicza macierz odległości euklidesowych.
    A: (Na, 2), B: (Nb, 2) -> Wynik: (Na, Nb)
    """
    return cdist(A, B, metric='euclidean')

def dist_sq(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Kwadrat odległości (szybsze do prostych porównań)."""
    return cdist(A, B, metric='sqeuclidean')