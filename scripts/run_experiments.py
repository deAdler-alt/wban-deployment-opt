import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Dodaj src do ścieżki (dla pewności)
sys.path.append(os.path.join(os.getcwd(), 'src'))

from wban_opt.config import load_points, load_gw_positions, load_scenarios
from wban_opt.geometry import points_to_numpy
from wban_opt.energy_model import EnergyParams
from wban_opt.objective import ObjectiveContext, energy_and_penalty_from_x
from wban_opt.mealpy_runner import solve_ga, solve_pso
from wban_opt.metrics import feasible_rate

# ---- HELPER: Generowanie dummy danych jeśli brak ----
def ensure_data_exists():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    points_path = data_dir / "points_body.csv"
    if not points_path.exists():
        print("WARNING: 'points_body.csv' not found. Generating dummy grid...")
        with open(points_path, 'w') as f:
            f.write("id,name,x,y,region\n")
            cnt = 0
            # Siatka 5x10 na obszarze 0.6x1.8
            for ix in range(5):
                for iy in range(10):
                    x = ix * (0.6/4)
                    y = iy * (1.8/9)
                    f.write(f"{cnt},P_{cnt},{x:.2f},{y:.2f},torso\n")
                    cnt += 1
        print(f"Generated {cnt} dummy points.")

def main():
    ensure_data_exists()
    
    # 1. Setup
    print("Loading configuration...")
    points = load_points("data/points_body.csv")
    points_pool_np = points_to_numpy(points)
    gw_positions = load_gw_positions("data/gw_positions.json")
    config = load_scenarios("data/scenarios.yaml")
    
    common = config['common']
    ep_cfg = common['energy']
    energy_params = EnergyParams(
        E_elec=float(ep_cfg['E_elec']),
        E_fs=float(ep_cfg['E_fs']),
        E_mp=float(ep_cfg['E_mp']),
        E_agg=float(ep_cfg['E_agg']),
        packet_bits=int(ep_cfg['packet_bits']),
        beta_agg=float(ep_cfg['beta_agg'])
    )
    
    # 2. Results container
    results = []
    
    # 3. Main Loop
    for scen in config['scenarios']:
        s_name = scen['name']
        N = scen['N']
        K = scen['K']
        M_limit = scen['M']
        D = N + K
        
        # Filtruj punkty (pierwsze M)
        if M_limit > len(points_pool_np):
            raise ValueError(f"Scenario {s_name} requires {M_limit} points, but only {len(points_pool_np)} available.")
        
        current_points_pool = points_pool_np[:M_limit]
        
        if D > M_limit:
            print(f"SKIPPING {s_name}: D={D} > M={M_limit} (Impossible unique placement)")
            continue
            
        print(f"--- Running Scenario: {s_name} (N={N}, K={K}, M={M_limit}) ---")
        
        for gw_name in scen['gw_variants']:
            gw_xy = np.array([gw_positions[gw_name]])
            
            # Context
            ctx = ObjectiveContext(
                points_pool=current_points_pool,
                N=N, K=K, gw_xy=gw_xy,
                energy_params=energy_params,
                d_max_sn_ch=float(common['constraints']['d_max_sn_ch']),
                penalty_weight=float(common['constraints']['penalty_range'])
            )
            
            n_runs = common['optimization']['n_runs']
            base_seed = common['optimization']['seed0']
            
            for r in range(n_runs):
                current_seed = base_seed + r
                
                # --- BASELINE (Random) ---
                np.random.seed(current_seed)
                x_rnd = np.random.rand(D)
                e_base, p_base, f_base = energy_and_penalty_from_x(x_rnd, ctx)
                
                # --- GA ---
                # Uwaga: w Mealpy seed wpływa na inicjalizację
                sol_ga, fit_ga_mealpy = solve_ga(
                    ctx, 
                    epochs=common['optimization']['epochs'], 
                    pop_size=common['optimization']['pop_size'],
                    seed=current_seed
                )
                e_ga, p_ga, f_ga_bool = energy_and_penalty_from_x(sol_ga, ctx)
                
                # --- PSO ---
                sol_pso, fit_pso_mealpy = solve_pso(
                    ctx, 
                    epochs=common['optimization']['epochs'], 
                    pop_size=common['optimization']['pop_size'],
                    seed=current_seed
                )
                e_pso, p_pso, f_pso_bool = energy_and_penalty_from_x(sol_pso, ctx)
                
                # Record
                row = {
                    "scenario": s_name,
                    "gw": gw_name,
                    "run": r,
                    "seed": current_seed,
                    "N": N, "K": K, "M": M_limit,
                    
                    "base_E": e_base, "base_P": p_base, "base_feas": f_base,
                    
                    "ga_E": e_ga, "ga_P": p_ga, "ga_feas": f_ga_bool,
                    "ga_fitness_mealpy": fit_ga_mealpy,
                    
                    "pso_E": e_pso, "pso_P": p_pso, "pso_feas": f_pso_bool,
                    "pso_fitness_mealpy": fit_pso_mealpy
                }
                results.append(row)
                
                print(f"   Run {r}: GA_E={e_ga:.4f} (OK={f_ga_bool}), PSO_E={e_pso:.4f} (OK={f_pso_bool})")

    # 4. Save
    out_dir = Path("results/csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "runs.csv", index=False)
    
    print("\n=== Summary ===")
    print(df.groupby(['scenario', 'gw'])[['ga_feas', 'pso_feas']].mean())
    print(f"Results saved to {out_dir / 'runs.csv'}")

if __name__ == "__main__":
    main()