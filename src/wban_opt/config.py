import pandas as pd
import json
import yaml
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Point:
    id: int
    name: str
    x: float
    y: float
    region: str

def load_points(path: str) -> list[Point]:
    df = pd.read_csv(path)
    # Walidacja kolumn
    required = {'id', 'name', 'x', 'y', 'region'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV missing columns: {required - set(df.columns)}")
    
    points = []
    for _, row in df.iterrows():
        points.append(Point(
            id=int(row['id']),
            name=str(row['name']),
            x=float(row['x']),
            y=float(row['y']),
            region=str(row['region'])
        ))
    return points

def load_gw_positions(path: str) -> dict[str, tuple[float, float]]:
    with open(path, 'r') as f:
        data = json.load(f)
    return {k: tuple(v) for k, v in data.items()}

def load_scenarios(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)