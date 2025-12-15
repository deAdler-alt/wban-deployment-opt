for sc in cfg["scenarios"]:
    name = sc["name"]
    N = int(sc["N"])
    K = int(sc["K"])
    M_sc = int(sc["M"])

    # weź tylko pierwsze M_sc punktów (po ID w CSV)
    points_sc = points[:M_sc]

    for gw_name in sc["gw_variants"]:
        gw_xy = gw_map[gw_name]

        ctx = ObjectiveContext(
            points=points_sc,
            N=N,
            K=K,
            gw_xy=gw_xy,
            energy=energy,
            d_max_sn_ch=float(cons_cfg["d_max_sn_ch"]),
            penalty_range=float(cons_cfg["penalty_range"]),
        )

        M = len(points_sc)
        D = N + K
        if D > M:
            raise ValueError(f"Scenariusz {name}/{gw_name} ma D={D} > M={M}. Zwiększ M lub zmniejsz N+K.")
