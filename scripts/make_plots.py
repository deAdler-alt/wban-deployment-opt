import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    csv_path = Path("results/csv/runs.csv")
    if not csv_path.exists():
        print("No results found. Run scripts/run_experiments.py first.")
        return
        
    df = pd.read_csv(csv_path)
    plots_dir = Path("results/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Przykładowy wykres: Boxplot energii (tylko dla rozwiązań feasible)
    # Filtrujemy tylko te, gdzie P=0 (lub bardzo małe)
    df_ok = df[ (df['ga_feas'] == True) & (df['pso_feas'] == True) ]
    
    if df_ok.empty:
        print("No feasible solutions to plot energy comparison.")
    else:
        # Pivot do boxplota
        # Chcemy porównać GA vs PSO per scenariusz
        # Uproszczenie: bierzemy pierwszy wariant GW
        first_gw = df_ok['gw'].unique()[0]
        sub = df_ok[df_ok['gw'] == first_gw]
        
        scenarios = sub['scenario'].unique()
        data_ga = []
        data_pso = []
        labels = []
        
        for s in scenarios:
            d_s = sub[sub['scenario'] == s]
            data_ga.append(d_s['ga_E'].values)
            data_pso.append(d_s['pso_E'].values)
            labels.append(s)
            
        # Rysowanie
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Przesunięcia
        pos_ga = [x - 0.2 for x in range(len(labels))]
        pos_pso = [x + 0.2 for x in range(len(labels))]
        
        bp1 = ax.boxplot(data_ga, positions=pos_ga, widths=0.3, patch_artist=True)
        bp2 = ax.boxplot(data_pso, positions=pos_pso, widths=0.3, patch_artist=True)
        
        # Kolory
        for patch in bp1['boxes']: patch.set_facecolor('lightblue')
        for patch in bp2['boxes']: patch.set_facecolor('lightgreen')
            
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_title(f"Energy Comparison (GW={first_gw})")
        ax.set_ylabel("Energy [J]")
        ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['GA', 'PSO'])
        
        out_file = plots_dir / "energy_boxplot.png"
        plt.savefig(out_file)
        print(f"Plot saved: {out_file}")

if __name__ == "__main__":
    main()