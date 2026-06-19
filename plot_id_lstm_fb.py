"""
Compare standard LSTM vs forget-bias LSTM on IntervalDiscrimination-v0.

For each num_layers, plots the best training curve (best sustained reward
across hidden_dim / lr runs) for both conditions.

Usage:
    python plot_id_lstm_fb.py [output]
"""

import sys

import duckdb
import matplotlib.pyplot as plt
import numpy as np

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "id_lstm_fb.png"

ENV = "IntervalDiscrimination-v0"
LAYERS = [1, 2, 3]

CONDITIONS = {
    "LSTM (standard)": ("ng-sweep-v1", "#1f77b4"),
    "LSTM (forget bias)": ("lstm-fb-sweep-v1", "#ff7f0e"),
}


def smooth(values, window=10):
    if window <= 1 or len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


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

all_tags = ", ".join(f"'{t}'" for _, (t, _) in CONDITIONS.items())

df = con.execute(f"""
    WITH windowed AS (
        SELECT
            experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name
                ORDER BY episode
                ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
            ) AS rolling_reward
        FROM metrics
    ),
    rolled AS (
        SELECT experiment_name, MAX(rolling_reward) AS best_sustained
        FROM windowed
        GROUP BY experiment_name
    ),
    ranked AS (
        SELECT
            c.tag, c.num_layers, c.experiment_name,
            ROW_NUMBER() OVER (
                PARTITION BY c.tag, c.num_layers
                ORDER BY r.best_sustained DESC
            ) AS rn
        FROM configs c
        JOIN rolled r ON c.experiment_name = r.experiment_name
        WHERE c.env_name = '{ENV}'
          AND c.model_type = 'lstm'
          AND c.tag IN ({all_tags})
    ),
    best AS (
        SELECT tag, num_layers, experiment_name FROM ranked WHERE rn = 1
    )
    SELECT b.tag, b.num_layers, b.experiment_name, m.episode, m.reward
    FROM best b
    JOIN metrics m ON b.experiment_name = m.experiment_name
    ORDER BY b.tag, b.num_layers, m.episode
""").fetchdf()
con.close()

fig, axes = plt.subplots(1, len(LAYERS), figsize=(5 * len(LAYERS), 4), sharey=True)
fig.suptitle(f"LSTM Forget-Bias Ablation — {ENV}", fontsize=13, fontweight="bold")

for col, nl in enumerate(LAYERS):
    ax = axes[col]
    ax.set_title(f"{nl} layer{'s' if nl > 1 else ''}", fontsize=11)
    has_any = False

    for label, (tag, color) in CONDITIONS.items():
        cell = df[(df["tag"] == tag) & (df["num_layers"] == nl)].sort_values("episode")
        if cell.empty:
            continue
        has_any = True
        ep = cell["episode"].values
        rew = cell["reward"].values
        sm = smooth(rew)
        ep_sm = ep[len(ep) - len(sm) :]
        ax.plot(ep_sm, sm, color=color, linewidth=2, label=label)

    if not has_any:
        ax.text(
            0.5,
            0.5,
            "no data yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="gray",
        )

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
    ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("Episode", fontsize=9)
    ax.grid(True, alpha=0.2)
    if col == 0:
        ax.set_ylabel("Avg Reward (rolling 10)", fontsize=9)

handles, labels = axes[0].get_legend_handles_labels()
if not handles:
    handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right", fontsize=9, framealpha=0.8)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
