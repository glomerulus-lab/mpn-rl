"""
Bar plot of highest average reward per environment for the custom sweep.

For each environment, shows one bar per model type at the mean peak reward of
the best hyperparameter config (selected by mean across sweep versions), with
error bars showing ±1 std across those runs. Duplicate runs within a sweep
version are resolved by keeping the most complete run (highest max frame).

Usage:
    python plot_custom_sweep_bar.py [sweeps] [output]

    sweeps - comma-separated sweep names (default: ng-sweep-v1)
    output - output file path (default: custom_sweep_bar.png)
"""

import sys

import duckdb
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

_sweeps_arg = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "ng-sweep-v1,ng-sweep-v2,ng-sweep-v3,ng-sweep-v4"
)
SWEEPS = [t.strip() for t in _sweeps_arg.split(",")]
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "custom_sweep_bar.png"

MODEL_TYPES = ["mpn-frozen", "lstm", "mpn", "rnn"]

MODEL_COLORS = {
    "lstm": "#1f77b4",
    "mpn": "#2ca02c",
    "mpn-frozen": "#d62728",
    "rnn": "#ff7f0e",
}


def lighten(color, factor=0.5):
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * factor, g + (1 - g) * factor, b + (1 - b) * factor)


def darken(color, factor=0.4):
    r, g, b = mcolors.to_rgb(color)
    return (r * (1 - factor), g * (1 - factor), b * (1 - factor))


LONG_LABEL_BREAKS = {
    "PerceptualDecisionMakingDelayResponse-v0": "PerceptualDecisionMaking\nDelayResponse",
}

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

tags_sql = ", ".join(f"'{t}'" for t in SWEEPS)

df = con.execute(f"""
    WITH windowed AS (
        SELECT
            experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name
                ORDER BY frame
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS reward_window,
            MAX(frame) OVER (PARTITION BY experiment_name) AS max_frame
        FROM metrics
    ),
    run_peaks AS (
        SELECT
            c.experiment_name,
            c.sweep_name,
            c.env_name,
            c.model_type,
            c.hidden_dim,
            c.num_layers,
            c.learning_rate,
            MAX(w.max_frame)      AS max_frame,
            MAX(w.reward_window)  AS peak_reward
        FROM configs c
        JOIN windowed w ON c.experiment_name = w.experiment_name
        WHERE c.sweep_name IN ({tags_sql})
          AND c.env_name LIKE '%-v0'
        GROUP BY
            c.experiment_name, c.sweep_name, c.env_name, c.model_type,
            c.hidden_dim, c.num_layers, c.learning_rate
    ),
    deduped AS (
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY sweep_name, env_name, model_type, hidden_dim, num_layers, learning_rate
                       ORDER BY max_frame DESC
                   ) AS rn
            FROM run_peaks
        )
        WHERE rn = 1
    ),
    best_per_tag AS (
        SELECT * EXCLUDE (rn)
        FROM (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY sweep_name, env_name, model_type
                       ORDER BY peak_reward DESC
                   ) AS rn
            FROM deduped
        )
        WHERE rn = 1
    )
    SELECT
        env_name, model_type,
        AVG(peak_reward)    AS mean_reward,
        STDDEV(peak_reward) AS std_reward
    FROM best_per_tag
    GROUP BY env_name, model_type
    ORDER BY env_name, model_type
""").fetchdf()
con.close()

ENVS = sorted(df["env_name"].unique())

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

n_envs = len(ENVS)
n_models = len(MODEL_TYPES)
bar_w = 0.12
gap = 0.05
offsets = np.arange(n_models) * bar_w - (n_models - 1) * bar_w / 2
group_x = np.arange(n_envs) * (n_models * bar_w + gap)

fig, ax = plt.subplots(figsize=(10, 4))

for j, mt in enumerate(MODEL_TYPES):
    color = MODEL_COLORS[mt]
    for i, env in enumerate(ENVS):
        row_data = df[(df["env_name"] == env) & (df["model_type"] == mt)]
        has_data = not row_data.empty
        xpos = group_x[i] + offsets[j]
        if has_data:
            mean = float(row_data["mean_reward"].iloc[0])
            std = float(row_data["std_reward"].iloc[0])
            std = std if not np.isnan(std) else 0.0
            ax.bar(xpos, mean, width=bar_w, color=color, zorder=3)
            ax.errorbar(
                xpos,
                mean,
                yerr=std,
                fmt="none",
                color=darken(color),
                capsize=1.5,
                linewidth=1.2,
                zorder=4,
            )
        else:
            ax.bar(
                xpos,
                0.015,
                width=bar_w,
                color="none",
                edgecolor=color,
                linewidth=1.0,
                linestyle="--",
                zorder=3,
            )

ax.set_xticks(group_x)
ax.set_xticklabels(
    [LONG_LABEL_BREAKS.get(e, e.replace("-v0", "")) for e in ENVS],
    rotation=40,
    ha="right",
    rotation_mode="anchor",
    fontsize=9,
)
ax.set_xlim(group_x[0] - 0.4, group_x[-1] + 0.4)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Best Avg Reward", fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.axhline(0, color="gray", linewidth=0.6)

handles = [plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[mt]) for mt in MODEL_TYPES]
ax.legend(
    handles,
    MODEL_TYPES,
    title="Model",
    fontsize=8,
    title_fontsize=8,
    loc="upper left",
    bbox_to_anchor=(1.01, 1),
    borderaxespad=0,
    framealpha=0.8,
)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
