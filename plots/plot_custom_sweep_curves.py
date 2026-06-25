"""
Plot training curves for hyperparameter sweep (ng-sweep-v1 by default).

Grid: rows = environments, columns = model types.
Within each cell: one curve per (num_layers, learning_rate), averaged over hidden_dim.
Color shade = num_layers (light → dark), line style = learning_rate (solid/dashed).

Usage:
    python plot_custom_sweep_curves.py [sweep] [output]
"""

import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpn_rl.experiment import find_experiment_files, load_experiments

_sweeps_arg = sys.argv[1] if len(sys.argv) > 1 else "ng-sweep-v1"
SWEEPS = [t.strip() for t in _sweeps_arg.split(",")]
SWEEP = _sweeps_arg
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "custom_sweep_curves.png"

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

MODEL_TYPES = ["mpn-frozen", "lstm", "mpn", "rnn"]

MODEL_BASE_COLORS = {
    "lstm": "#1f77b4",
    "mpn": "#2ca02c",
    "mpn-frozen": "#d62728",
    "rnn": "#ff7f0e",
}


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(values).rolling(window, min_periods=1).mean().values


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

con = duckdb.connect()
metrics_list = (
    "["
    + ", ".join(f"'{p}'" for p in find_experiment_files("metrics.jsonl", None))
    + "]"
)
con.execute(f"""
    CREATE VIEW metrics AS
    SELECT experiment_name, episode, reward
    FROM read_ndjson(
        {metrics_list},
        columns = {{experiment_name: 'VARCHAR', episode: 'INTEGER', reward: 'DOUBLE'}},
        ignore_errors = true
    )
""")
con.register("configs", load_experiments())

df = con.execute(f"""
    WITH rolling AS (
        SELECT
            c.experiment_name, c.sweep_name, c.env_name, c.model_type,
            c.hidden_dim, c.num_layers, c.learning_rate,
            m.episode,
            AVG(m.reward) OVER (
                PARTITION BY c.experiment_name
                ORDER BY m.episode
                ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS rolling_reward
        FROM configs c
        JOIN metrics m ON c.experiment_name = m.experiment_name
        WHERE c.sweep_name IN ({", ".join(f"'{t}'" for t in SWEEPS)})
          AND c.env_name LIKE '%-v0'
    ),
    run_peaks AS (
        SELECT
            experiment_name, sweep_name, env_name, model_type,
            hidden_dim, num_layers, learning_rate,
            MAX(episode)        AS max_episode,
            MAX(rolling_reward) AS peak_reward
        FROM rolling
        GROUP BY
            experiment_name, sweep_name, env_name, model_type,
            hidden_dim, num_layers, learning_rate
    ),
    deduped AS (
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY sweep_name, env_name, model_type, hidden_dim, num_layers, learning_rate
                    ORDER BY max_episode DESC
                ) AS rn
            FROM run_peaks
        ) WHERE rn = 1
    ),
    best_per_sweep AS (
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY sweep_name, env_name, model_type
                    ORDER BY peak_reward DESC
                ) AS rn
            FROM deduped
        ) WHERE rn = 1
    )
    SELECT
        b.env_name, b.model_type, b.sweep_name, b.experiment_name,
        m.episode, m.reward
    FROM best_per_sweep b
    JOIN metrics m ON b.experiment_name = m.experiment_name
    ORDER BY b.env_name, b.model_type, b.sweep_name, m.episode
""").fetchdf()
con.close()

if df.empty:
    print(f"No data found for sweep='{SWEEP}'.")
    sys.exit(1)

envs = sorted(df["env_name"].unique())
n_rows = len(envs)
n_cols = len(MODEL_TYPES)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(4.5 * n_cols, 3.5 * n_rows),
    squeeze=False,
    sharey="row",
)

fig.suptitle(f"Training Curves — {SWEEP}", fontsize=13, fontweight="bold", y=1.01)

smooth_window = 50

for col, mt in enumerate(MODEL_TYPES):
    axes[0, col].set_title(mt, fontsize=11, fontweight="bold", pad=6)

for row, env in enumerate(envs):
    for col, mt in enumerate(MODEL_TYPES):
        ax = axes[row, col]
        cell = df[(df["env_name"] == env) & (df["model_type"] == mt)]

        if cell.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="gray",
            )
            ax.set_ylim(-0.15, 1.05)
            ax.grid(True, alpha=0.2)
            continue

        color = MODEL_BASE_COLORS.get(mt, "#888888")

        sweep_curves = []
        for _, sweep_grp in cell.groupby("sweep_name"):
            sweep_grp = sweep_grp.sort_values("episode")
            sm = smooth(sweep_grp["reward"].values, smooth_window)
            sweep_curves.append((sweep_grp["episode"].values, sm))

        if len(sweep_curves) == 1:
            ep, sm = sweep_curves[0]
            ax.plot(ep, sm, color=color, linewidth=1.8, zorder=3)
        else:
            max_ep = max(c[0][-1] for c in sweep_curves)
            common_ep = np.linspace(0, max_ep, 500)
            stack = np.array(
                [
                    np.interp(common_ep, ep, sm, left=np.nan, right=np.nan)
                    for ep, sm in sweep_curves
                ]
            )
            mean_curve = np.nanmean(stack, axis=0)
            std_curve = np.nanstd(stack, axis=0)
            valid = ~np.isnan(mean_curve)
            ax.plot(
                common_ep[valid],
                mean_curve[valid],
                color=color,
                linewidth=1.8,
                zorder=3,
            )
            ax.fill_between(
                common_ep[valid],
                (mean_curve - std_curve)[valid],
                (mean_curve + std_curve)[valid],
                color=color,
                alpha=0.2,
                zorder=2,
            )

        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_ylim(-0.15, 1.05)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=7)
        if row == n_rows - 1:
            ax.set_xlabel("Episode", fontsize=8)

    axes[row, 0].set_ylabel(env, fontsize=9, fontweight="bold", labelpad=8)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
