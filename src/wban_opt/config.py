from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any
import csv
import json

try:
    import yaml
except ImportError as e:
    raise ImportError("Brakuje PyYAML. Dodaj do requirements.txt: pyyaml") from e


@dataclass(frozen=True)
class Point:
    id: int
    name: str
    x: float
    y: float
    region: str


def load_points(path: str | Path) -> List[Point]:
    path = Path(path)
    points: List[Point] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "name", "x", "y", "region"}
        if set(reader.fieldnames or []) < required:
            raise ValueError(f"CSV {path} musi mieć kolumny: {sorted(required)}")

        for row in reader:
            p = Point(
                id=int(row["id"]),
                name=str(row["name"]),
                x=float(row["x"]),
                y=float(row["y"]),
                region=str(row["region"]),
            )
            points.append(p)

    # sortowanie po id (ważne, bo scenariusze używają M jako "pierwsze M punktów")
    points.sort(key=lambda p: p.id)
    return points


def load_gw_positions(path: str | Path) -> Dict[str, Tuple[float, float]]:
    path = Path(path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    out: Dict[str, Tuple[float, float]] = {}
    for k, v in obj.items():
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError(f"GW {k} musi mieć listę [x,y], jest: {v}")
        out[str(k)] = (float(v[0]), float(v[1]))
    return out


def load_scenarios(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("scenarios.yaml musi być słownikiem na poziomie root")
    return cfg
