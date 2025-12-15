from __future__ import annotations

from typing import List, Tuple
import numpy as np

from .config import Point


def points_xy(points: List[Point]) -> np.ndarray:
    """
    Zamienia listę Point na macierz współrzędnych (M x 2).
    M = liczba punktów montażu.
    """
    return np.asarray([(p.x, p.y) for p in points], dtype=float)


def euclid(a: np.ndarray, b: np.ndarray) -> float:
    """
    Odległość euklidesowa 2D pomiędzy dwoma punktami (2,).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Zwraca macierz odległości pomiędzy dwoma zbiorami punktów:
    A: (n x 2), B: (m x 2) -> (n x m)
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    diff = A[:, None, :] - B[None, :, :]
    return np.linalg.norm(diff, axis=2)
