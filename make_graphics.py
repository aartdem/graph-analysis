import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from csv import DictReader
from matplotlib.ticker import LogLocator, LogFormatter

graphs_metadata_path = "graphs_metadata.csv"


def load_graph_stats():
    result = {}
    with Path(graphs_metadata_path).expanduser().open(newline="", encoding="utf-8") as fh:
        reader = DictReader(fh)
        expected = {"Graph", "Vertexes", "Edges"}
        if set(reader.fieldnames or []) != expected:
            raise ValueError(
                f"CSV must have header exactly: {', '.join(expected)}"
            )

        for row in reader:
            name = row["Graph"].strip()
            try:
                v = float(row["Vertexes"])
                e = float(row["Edges"])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in row {reader.line_num}: {row}"
                ) from exc
            result[name] = (v, e)

    return result


def plot_comparison(df, algos, title, output_filename, color_map, ylim=None, ci_level=0.95):
    graphs_stats = load_graph_stats()

    graphs = sorted(df["Graph"].unique(), key=lambda name: (
        graphs_stats[name][0],
        graphs_stats[name][1],
        name
    ))
    alpha = 1 - ci_level

    # Prepare data
    means = []
    cis = []
    for algo in algos:
        sub = df[df['Algorithm'] == algo]
        grp = sub.groupby('Graph')['time']
        mean = grp.mean().reindex(graphs)
        std = grp.std(ddof=1).reindex(graphs)
        n = grp.count().reindex(graphs)
        sem = std / np.sqrt(n)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci = sem * t_crit
        means.append(mean.values)
        cis.append(ci.values)

    # Plot
    x = np.arange(len(graphs))
    width = 0.8 / len(algos)  # Adjust width based on number of algorithms
    fig, ax = plt.subplots()
    error_kwargs = dict(elinewidth=1.5, capsize=5)

    for i, algo in enumerate(algos):
        offset = (i - (len(algos) - 1) / 2) * width
        ax.bar(x + offset, means[i], width, yerr=cis[i], error_kw=error_kwargs,
               label=algo, color=color_map.get(algo))

    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=45, ha='right')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(-0.01, 1.03))  # Move slightly higher and to the left
    ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth=1, color="#555555", alpha=0.95)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=600)
    plt.close()


def main():
    # Load and transform data
    raw = pd.read_csv('benchmark_results_all.csv')
    time_cols = [col for col in raw.columns if col not in ['Algorithm', 'Graph']]
    df = raw.melt(id_vars=['Algorithm', 'Graph'], value_vars=time_cols,
                  var_name='run', value_name='time')
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['time'])

    # Convert milliseconds to seconds
    df['time'] = df['time'] / 1000.0

    # Define color mapping: Gunrock always blue, Spla orange/green, Lagraph red
    color_map = {
        'BoruvkaGunrock': 'tab:blue',
        'PrimGunrock': 'tab:blue',
        'BoruvkaSpla': 'tab:orange',
        'BoruvkaSplaGpu': 'tab:orange',
        'PrimSpla': 'tab:orange',
        'BoruvkaSplaCpu': 'tab:green',
        'BoruvkaLagraph': 'tab:red'
    }

    # Define comparisons
    comparisons = [
        # (['BoruvkaSplaGpu', 'BoruvkaSplaCpu', 'BoruvkaGunrock', 'BoruvkaLagraph'],
        #  'Boruvka: Spla GPU vs Spla CPU vs Gunrock GPU vs Lagraph CPU', 'comparison_boruvka.png'),
        # (['PrimSpla', 'PrimGunrock'], 'Prim: Spla CPU (no OpenCL) vs Gunrock GPU', 'comparison_prim.png'),
        (['BfsSpla', 'BfsLagraph'], 'Parent Bfs: Spla CPU (no OpenCL) vs LaGraph', 'comparison_bfs.png')
    ]

    # Compute common y-limits across all comparisons
    all_max = 0.0
    all_min = np.inf
    for algos, _, _ in comparisons:
        for algo in algos:
            sub = df[df['Algorithm'] == algo]
            grp = sub.groupby('Graph')['time']
            mean = grp.mean()
            std = grp.std(ddof=1)
            n = grp.count()
            sem = std / np.sqrt(n)
            # approximate CI max using z-value
            upper = (mean + sem * stats.norm.ppf(0.975)).max()
            all_max = max(all_max, upper)
    all_min = df['time'].min()
    ylim = (max(all_min * 0.9, 1e-6), all_max * 1.1)

    # Generate plots with 95% CI
    for algos, title, filename in comparisons:
        plot_comparison(
            df,
            algos=algos,
            title=title,
            output_filename=filename,
            color_map=color_map,
            ylim=ylim,
            ci_level=0.95
        )


if __name__ == '__main__':
    main()
