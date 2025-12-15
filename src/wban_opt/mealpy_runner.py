from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from mealpy import FloatVar, GA, PSO

from .objective import objective_from_x


@dataclass(frozen=True)
class SolveResult:
    best_x: np.ndarray
    best_f: float
    algo: str


def _make_problem(ctx: Any) -> Dict[str, Any]:
    """
    Buduje problem dla Mealpy 3.x na przestrzeni ciągłej [0,1]^D.
    D = N + K (zgodnie z założeniami pracy).
    """
    if not hasattr(ctx, "N") or not hasattr(ctx, "K"):
        raise TypeError("ctx musi mieć atrybuty N i K")

    D = int(ctx.N) + int(ctx.K)

    return {
        "bounds": FloatVar(lb=(0.0,) * D, ub=(1.0,) * D, name="x"),
        "obj_func": lambda sol: objective_from_x(sol, ctx),
        "minmax": "min",
    }


def solve_ga(
    ctx: Any,
    *,
    epochs: int,
    pop_size: int,
    pc: float = 0.9,
    pm: float = 0.1,
    selection: str = "tournament",
    k_way: float = 0.2,
    crossover: str = "uniform",
    mutation: str = "swap",
) -> SolveResult:
    """
    GA z Mealpy: działa na x∈[0,1]^D.
    Dekodowanie do indeksów punktów + unikalność robimy w objective.py (decode+repair).
    """
    problem = _make_problem(ctx)

    model = GA.BaseGA(
        epoch=int(epochs),
        pop_size=int(pop_size),
        pc=float(pc),
        pm=float(pm),
        selection=str(selection),
        k_way=float(k_way),
        crossover=str(crossover),
        mutation=str(mutation),
    )

    g_best = model.solve(problem)
    return SolveResult(
        best_x=np.asarray(g_best.solution, dtype=float),
        best_f=float(g_best.target.fitness),
        algo="GA.BaseGA",
    )


def solve_pso(
    ctx: Any,
    *,
    epochs: int,
    pop_size: int,
    c1: float = 2.05,
    c2: float = 2.05,
    w: float = 0.4,
) -> SolveResult:
    """
    PSO z Mealpy (OriginalPSO).
    """
    problem = _make_problem(ctx)

    model = PSO.OriginalPSO(
        epoch=int(epochs),
        pop_size=int(pop_size),
        c1=float(c1),
        c2=float(c2),
        w=float(w),
    )

    g_best = model.solve(problem)
    return SolveResult(
        best_x=np.asarray(g_best.solution, dtype=float),
        best_f=float(g_best.target.fitness),
        algo="PSO.OriginalPSO",
    )
