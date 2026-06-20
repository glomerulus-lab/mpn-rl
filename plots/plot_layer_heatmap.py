"""
Heatmap of best average reward for a model type by num_layers and hidden_dim.

Layout: one subplot per environment arranged in a grid.
Within each subplot: rows = hidden_dim, columns = num_layers.
Cell color = best rolling-avg reward across all runs for that (env, num_layers, hidden_dim).

Usage:
    python plot_layer_heatmap.py [tag] [output] [model_type]

    tag        - experiment tag (default: ng-sweep-v1)
    output     - output file path (default: layer_heatmap.png)
    model_type - model to plot (default: mpn)
"""

import math
import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np

_tags_arg = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "ng-sweep-v1,ng-sweep-v2,ng-sweep-v3,ng-sweep-v4"
)
TAGS = [t.strip() for t in _tags_arg.split(",")]
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "layer_heatmap.png"
MODEL_TYPE = sys.argv[3] if len(sys.argv) > 3 else "mpn"

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

con = duckdb.connect()
con.execute("""
    CREATE VIEW metrics AS
    SELECT experiment_name, frame, reward
    FROM read_ndjson(
        'experiments/*/metrics.jsonl',
        columns = {experiment_name: 'VARCHAR', frame: 'INTEGER', reward: 'DOUBLE'},
        ignore_errors = true
    )
""")
con.execute("""
    CREATE VIEW configs AS
    SELECT * FROM read_json_auto('experiments/*/config.json', ignore_errors = true)
""")

tags_sql = ", ".join(f"'{t}'" for t in TAGS)

df = con.execute(f"""
    WITH windowed AS (
        SELECT
            experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name
                ORDER BY frame
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS reward_50,
            MAX(frame) OVER (PARTITION BY experiment_name) AS max_frame
        FROM metrics
    ),
    run_peaks AS (
        SELECT
            c.experiment_name, c.tag, c.env_name,
            c.num_layers, c.hidden_dim,
            MAX(w.reward_50) AS peak_reward
        FROM configs c
        JOIN windowed w ON c.experiment_name = w.experiment_name
        WHERE c.tag IN ({tags_sql})
          AND c.model_type = '{MODEL_TYPE}'
          AND c.num_layers IS NOT NULL
          AND c.hidden_dim IS NOT NULL
        GROUP BY
            c.experiment_name, c.tag, c.env_name,
            c.num_layers, c.hidden_dim
    )
    SELECT env_name, num_layers, hidden_dim, MAX(peak_reward) AS best_reward
    FROM run_peaks
    GROUP BY env_name, num_layers, hidden_dim
    ORDER BY env_name, num_layers, hidden_dim
""").fetchdf()
con.close()

ENVS = sorted(df["env_name"].unique())
LAYERS = sorted(df["num_layers"].dropna().unique().astype(int))
DIMS = sorted(
    df["hidden_dim"].dropna().unique().astype(int), reverse=True
)  # large at top

# ---------------------------------------------------------------------------
# Layout: grid of subplots, one per environment
# ---------------------------------------------------------------------------

n_cols_grid = 4
n_rows_grid = math.ceil(len(ENVS) / n_cols_grid)

cell_w = 2.2
cell_h = 1.8
fig, axes = plt.subplots(
    n_rows_grid,
    n_cols_grid,
    figsize=(
        cell_w * n_cols_grid * len(LAYERS) / 3 + 1,
        cell_h * n_rows_grid * len(DIMS) / 3 + 1.5,
    ),
    squeeze=False,
)

vmin, vmax = 0.0, 1.0
CMAPS = {"mpn": "Greens", "mpn-frozen": "Reds", "lstm": "Blues", "rnn": "Oranges"}
cmap = CMAPS.get(MODEL_TYPE, "viridis")

for idx, env in enumerate(ENVS):
    row, col = divmod(idx, n_cols_grid)
    ax = axes[row, col]

    sub = df[df["env_name"] == env]

    matrix = np.full((len(DIMS), len(LAYERS)), np.nan)
    for i, dim in enumerate(DIMS):
        for j, layer in enumerate(LAYERS):
            cell = sub[(sub["hidden_dim"] == dim) & (sub["num_layers"] == layer)]
            if not cell.empty:
                matrix[i, j] = float(cell["best_reward"].iloc[0])

    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    for i in range(len(DIMS)):
        for j in range(len(LAYERS)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if val > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )
            else:
                ax.text(
                    j, i, "—", ha="center", va="center", fontsize=8, color="#aaaaaa"
                )

    env_label = env.replace("-v0", "")
    # wrap long names so they don't overlap neighbouring subplots
    if len(env_label) > 22:
        mid = len(env_label) // 2
        # find nearest space or camel-case boundary to split on
        split = next(
            (i for i in range(mid, len(env_label)) if env_label[i].isupper()),
            mid,
        )
        env_label = env_label[:split] + "\n" + env_label[split:]
    ax.set_title(env_label, fontsize=8, fontweight="bold", pad=4, linespacing=1.2)
    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([str(layer) for layer in LAYERS], fontsize=8)
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels([str(d) for d in DIMS], fontsize=8)

    if row == n_rows_grid - 1 or idx == len(ENVS) - 1:
        ax.set_xlabel("Num layers", fontsize=8)
    if col == 0:
        ax.set_ylabel("Hidden dim", fontsize=8)

# Hide unused subplots
for idx in range(len(ENVS), n_rows_grid * n_cols_grid):
    row, col = divmod(idx, n_cols_grid)
    axes[row, col].set_visible(False)

# Shared colorbar
fig.subplots_adjust(right=0.88, hspace=0.85, wspace=0.35)
cbar_ax = fig.add_axes([0.91, 0.08, 0.013, 0.25])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
fig.colorbar(sm, cax=cbar_ax, label="Best avg reward")

plt.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
