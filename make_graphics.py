import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(df, algos, title, output_filename, color_map, ylim=None):
    graphs = sorted(df['Graph'].unique())
    
    # Prepare data
    means = []
    sems = []
    for algo in algos:
        sub = df[df['Algorithm'] == algo]
        grp = sub.groupby('Graph')['time']
        mean = grp.mean().reindex(graphs)
        sem = grp.std(ddof=1).reindex(graphs) / np.sqrt(grp.count().reindex(graphs))
        means.append(mean.values)
        sems.append(sem.values)
    
    # Plot
    x = np.arange(len(graphs))
    width = 0.35
    fig, ax = plt.subplots()
    error_kwargs = dict(elinewidth=1.5, capsize=5)
    # Plot each algorithm with specified color
    ax.bar(x - width/2, means[0], width, yerr=sems[0], error_kw=error_kwargs,
           label=algos[0], color=color_map.get(algos[0]))
    ax.bar(x + width/2, means[1], width, yerr=sems[1], error_kw=error_kwargs,
           label=algos[1], color=color_map.get(algos[1]))
    
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.set_title(title)
    ax.legend()
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()


def main():
    # Load and transform data
    raw = pd.read_csv('benchmark_results.csv')
    time_cols = [col for col in raw.columns if col not in ['Algorithm', 'Graph']]
    df = raw.melt(id_vars=['Algorithm', 'Graph'], value_vars=time_cols,
                  var_name='run', value_name='time')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])
    
    # Convert milliseconds to seconds
    df['time'] = df['time'] / 1000.0
    
    # Define color mapping: Gunrock always blue, Spla orange
    color_map = {
        'BoruvkaGunrock': 'tab:blue',
        'PrimGunrock': 'tab:blue',
        'BoruvkaSpla': 'tab:orange',
        'PrimSpla': 'tab:orange'
    }
    
    # Define comparisons
    comparisons = [
        (['BoruvkaGunrock', 'BoruvkaSpla'], 'Boruvka: Gunrock vs Spla', 'comparison_boruvka.png'),
        (['PrimSpla', 'PrimGunrock'], 'Prim: Spla vs Gunrock', 'comparison_prim.png')
    ]
    
    # Compute common y-limits across all comparisons in seconds
    all_max = 0.0
    all_min = np.inf
    for algos, _, _ in comparisons:
        for algo in algos:
            sub = df[df['Algorithm'] == algo]
            grp = sub.groupby('Graph')['time']
            mean = grp.mean()
            sem = grp.std(ddof=1) / np.sqrt(grp.count())
            upper = (mean + sem).max()
            all_max = max(all_max, upper)
    all_min = df['time'].min()
    ylim = (max(all_min * 0.9, 1e-6), all_max * 1.1)
    
    # Generate plots
    for algos, title, filename in comparisons:
        plot_comparison(
            df,
            algos=algos,
            title=title,
            output_filename=filename,
            color_map=color_map,
            ylim=ylim
        )


if __name__ == '__main__':
    main()

