import pandas as pd

def feasible_rate(series: pd.Series) -> float:
    """Zwraca odsetek (0-1) wartości True w serii."""
    if len(series) == 0:
        return 0.0
    return series.sum() / len(series)