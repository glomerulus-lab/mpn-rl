"""
Plot training curves for hyperparameter sweep (ng-sweep-v1 by default).

Grid: rows = environments, columns = model types.
Within each cell: one curve per (num_layers, learning_rate), averaged over hidden_dim.
Color shade = num_layers (light → dark), line style = learning_rate (solid/dashed).

Usage:
    python plot_custom_sweep_curves.py [tag] [output]
"""

import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_tags_arg = sys.argv[1] if len(sys.argv) > 1 else "ng-sweep-v1"
TAGS = [t.strip() for t in _tags_arg.split(",")]
TAG = _tags_arg
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
con.execute("""
    CREATE VIEW metrics AS
    SELECT experiment_name, episode, reward
    FROM read_ndjson(
        'experiments/*/metrics.jsonl',
        columns = {experiment_name: 'VARCHAR', episode: 'INTEGER', reward: 'DOUBLE'},
        ignore_errors = true
    )
""")
con.execute("""
    CREATE VIEW configs AS
    SELECT * FROM read_json_auto('experiments/*/config.json', ignore_errors = true)
""")

df = con.execute(f"""
    WITH run_peaks AS (
        SELECT
            c.experiment_name, c.tag, c.env_name, c.model_type,
            c.hidden_dim, c.num_layers, c.learning_rate,
            MAX(m.episode)  AS max_episode,
            MAX(AVG(m.reward) OVER (
                PARTITION BY c.experiment_name
                ORDER BY m.episode
                ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            )) AS peak_reward
        FROM configs c
        JOIN metrics m ON c.experiment_name = m.experiment_name
        WHERE c.tag IN ({", ".join(f"'{t}'" for t in TAGS)})
          AND c.env_name LIKE '%-v0'
        GROUP BY
            c.experiment_name, c.tag, c.env_name, c.model_type,
            c.hidden_dim, c.num_layers, c.learning_rate
    ),
    deduped AS (
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY tag, env_name, model_type, hidden_dim, num_layers, learning_rate
                    ORDER BY max_episode DESC
                ) AS rn
            FROM run_peaks
        ) WHERE rn = 1
    ),
    best_per_tag AS (
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY tag, env_name, model_type
                    ORDER BY peak_reward DESC
                ) AS rn
            FROM deduped
        ) WHERE rn = 1
    )
    SELECT
        b.env_name, b.model_type, b.tag, b.experiment_name,
        m.episode, m.reward
    FROM best_per_tag b
    JOIN metrics m ON b.experiment_name = m.experiment_name
    ORDER BY b.env_name, b.model_type, b.tag, m.episode
""").fetchdf()
con.close()

if df.empty:
    print(f"No data found for tag='{TAG}'.")
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

fig.suptitle(f"Training Curves — {TAG}", fontsize=13, fontweight="bold", y=1.01)

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

        tag_curves = []
        for _, tag_grp in cell.groupby("tag"):
            tag_grp = tag_grp.sort_values("episode")
            sm = smooth(tag_grp["reward"].values, smooth_window)
            tag_curves.append((tag_grp["episode"].values, sm))

        if len(tag_curves) == 1:
            ep, sm = tag_curves[0]
            ax.plot(ep, sm, color=color, linewidth=1.8, zorder=3)
        else:
            max_ep = max(c[0][-1] for c in tag_curves)
            common_ep = np.linspace(0, max_ep, 500)
            stack = np.array(
                [
                    np.interp(common_ep, ep, sm, left=np.nan, right=np.nan)
                    for ep, sm in tag_curves
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
