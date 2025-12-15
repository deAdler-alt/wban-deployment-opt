import numpy as np

def decode_to_indices(x: np.ndarray, M: int) -> np.ndarray:
    """
    Mapuje ciągły wektor x z [0, 1] na indeksy całkowite [0, M-1].
    """
    # Clip na wszelki wypadek (chociaż Mealpy trzyma bounds)
    x_clipped = np.clip(x, 0.0, 0.9999999)
    indices = np.floor(x_clipped * M).astype(int)
    return indices

def repair_unique(indices: np.ndarray, M: int, rng: np.random.Generator = None) -> np.ndarray:
    """
    Zapewnia unikalność indeksów w wektorze.
    Konflikty zastępuje losowymi wolnymi indeksami.
    """
    if rng is None:
        rng = np.random.default_rng()

    D = len(indices)
    if D > M:
        raise ValueError(f"Nie można wybrać {D} unikalnych z puli {M}!")

    unique_vals, counts = np.unique(indices, return_counts=True)
    
    # Jeśli mamy tyle unikalnych co długość wektora, jest OK
    if len(unique_vals) == D:
        return indices

    # Znajdź dostępne (wolne) indeksy
    all_indices = set(range(M))
    used_indices = set(unique_vals)
    available = list(all_indices - used_indices)
    rng.shuffle(available)

    # Naprawa
    result = indices.copy()
    seen = set()
    
    replace_ptr = 0
    for i in range(D):
        val = result[i]
        if val in seen:
            # Konflikt -> bierzemy z puli available
            result[i] = available[replace_ptr]
            replace_ptr += 1
        else:
            seen.add(val)
            
    return result