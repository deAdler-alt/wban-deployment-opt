import matplotlib.pyplot as plt
import numpy as np
from .config import Point

def plot_placement(points: list[Point], sn_idx: np.ndarray, ch_idx: np.ndarray, 
                   gw_xy: tuple[float, float], outpath: str):
    """
    Rysuje mapę ciała z zaznaczonymi węzłami.
    """
    plt.figure(figsize=(5, 8))
    
    # 1. Wszystkie punkty tła
    all_xy = np.array([[p.x, p.y] for p in points])
    plt.scatter(all_xy[:, 0], all_xy[:, 1], c='lightgray', s=30, label='Candidates')
    
    # 2. Sensory
    sn_xy = all_xy[sn_idx]
    plt.scatter(sn_xy[:, 0], sn_xy[:, 1], c='blue', s=80, marker='o', label='Sensor')
    
    # 3. CH
    ch_xy = all_xy[ch_idx]
    plt.scatter(ch_xy[:, 0], ch_xy[:, 1], c='red', s=100, marker='^', label='CH')
    
    # 4. Gateway
    plt.scatter([gw_xy[0]], [gw_xy[1]], c='green', s=150, marker='s', label='GW')
    
    # Ozdobniki
    plt.title("WBAN Placement")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    
    plt.savefig(outpath)
    plt.close()