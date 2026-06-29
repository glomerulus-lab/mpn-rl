"""
Plot a table of 2000-episode eval results for the best model per
(env, model_type) across the last 4 ng-sweep-v1 environments.

Usage:
    python plot_eval_table.py [output]
"""

import json
import sys
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mpn_rl.envs import _create_env_from_config
from mpn_rl.evaluation import evaluate_actorcritic
from mpn_rl.experiment import find_experiment_files, load_experiments
from mpn_rl.models.actor_critic import ActorCriticNet

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "eval_table.png"
SWEEP = sys.argv[2] if len(sys.argv) > 2 else "ng-sweep-v1"
NUM_EPISODES = 2000
DEVICE = torch.device("cpu")
MODEL_ORDER = ["lstm", "rnn", "mpn", "mpn-frozen"]

ENVS = [
    "MultiSensoryIntegration-v0",
    "PerceptualDecisionMaking-v0",
    "PerceptualDecisionMakingDelayResponse-v0",
    "ProbabilisticReasoning-v0",
]

ENV_LABELS = {
    "MultiSensoryIntegration-v0": "MultiSensoryIntegration",
    "PerceptualDecisionMaking-v0": "PerceptualDecisionMaking",
    "PerceptualDecisionMakingDelayResponse-v0": "PerceptualDecisionMaking\nDelayResponse",
    "ProbabilisticReasoning-v0": "ProbabilisticReasoning",
}

# ---------------------------------------------------------------------------
# Select best experiment per (env, model_type)
# ---------------------------------------------------------------------------

con = duckdb.connect()
metrics_list = (
    "["
    + ", ".join(f"'{p}'" for p in find_experiment_files("metrics.jsonl", None))
    + "]"
)
con.execute(f"""
    CREATE VIEW metrics AS
    SELECT experiment_name, frame, reward
    FROM read_ndjson(
        {metrics_list},
        columns = {{experiment_name: 'VARCHAR', frame: 'INTEGER', reward: 'DOUBLE'}},
        ignore_errors = true
    )
""")
con.register("configs", load_experiments())

env_filter = ", ".join(f"'{e}'" for e in ENVS)
best_df = con.execute(f"""
    WITH windowed AS (
        SELECT experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name ORDER BY frame
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS rolling_reward
        FROM metrics
    ),
    best_per_exp AS (
        SELECT experiment_name, MAX(rolling_reward) AS best_rolling
        FROM windowed GROUP BY experiment_name
    ),
    ranked AS (
        SELECT c.env_name, c.model_type, c.experiment_name, c.path, b.best_rolling,
            ROW_NUMBER() OVER (
                PARTITION BY c.env_name, c.model_type ORDER BY b.best_rolling DESC
            ) AS rn
        FROM configs c JOIN best_per_exp b ON c.experiment_name = b.experiment_name
        WHERE c.sweep_name = '{SWEEP}' AND c.env_name IN ({env_filter})
    )
    SELECT env_name, model_type, experiment_name, path
    FROM ranked WHERE rn = 1
""").fetchdf()
con.close()

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

# results[env][model_type] = (mean, std)
results = {env: {} for env in ENVS}

for _, row in best_df.iterrows():
    exp_name = row["experiment_name"]
    env_name = row["env_name"]
    model_type = row["model_type"]

    run_dir = Path(row["path"])
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    cfg_path = run_dir / "config.json"

    if not ckpt_path.exists():
        results[env_name][model_type] = (float("nan"), float("nan"))
        continue

    with open(cfg_path) as f:
        config = json.load(f)

    env_tmp = _create_env_from_config(config)
    model = ActorCriticNet(
        input_dim=env_tmp.observation_space.shape[0],
        action_dim=env_tmp.action_space.n,
        hidden_dim=config.get("hidden_dim", 128),
        model_type=config.get("model_type", "lstm"),
        activation=config.get("activation", "tanh"),
        lambda_max=config.get("lambda_max", 0.99),
        eta_init=config.get("eta_init", 0.01),
        lambda_init=config.get("lambda_init", 0.99),
        num_layers=config.get("num_layers", 1),
        mpn_bias=config.get("mpn_bias", True),
    ).to(DEVICE)
    env_tmp.close()

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Evaluating {model_type:12s} on {env_name}...")
    ep_rewards = evaluate_actorcritic(
        model,
        lambda cfg=config: _create_env_from_config(cfg),
        NUM_EPISODES,
        config.get("max_episode_steps", 500),
        seed=0,
        device=DEVICE,
    )
    mean_r, std_r = float(np.mean(ep_rewards)), float(np.std(ep_rewards))
    results[env_name][model_type] = (mean_r, std_r)
    print(f"  mean={mean_r:.4f}  std={std_r:.4f}")

# ---------------------------------------------------------------------------
# Build table data
# ---------------------------------------------------------------------------

col_labels = [m.upper().replace("-", "\n") for m in MODEL_ORDER]
row_labels = [ENV_LABELS[e] for e in ENVS]

cell_text = []
cell_colors = []

for env in ENVS:
    row_text = []
    row_colors = []
    means = [results[env].get(m, (float("nan"), float("nan")))[0] for m in MODEL_ORDER]
    best_mean = max((v for v in means if not np.isnan(v)), default=float("nan"))

    for mt, mean in zip(MODEL_ORDER, means):
        std = results[env].get(mt, (float("nan"), float("nan")))[1]
        if np.isnan(mean):
            row_text.append("—")
            row_colors.append("#f0f0f0")
        else:
            row_text.append(f"{mean:.3f}\n±{std:.3f}")
            if mean == best_mean:
                row_colors.append("#c6efce")  # green highlight for best
            elif mean < 0.1:
                row_colors.append("#ffc7ce")  # red for failed
            else:
                row_colors.append("#ffffff")
    cell_text.append(row_text)
    cell_colors.append(row_colors)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis("off")

tbl = ax.table(
    cellText=cell_text,
    cellColours=cell_colors,
    rowLabels=row_labels,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.2, 2.2)

ax.set_title(
    f"Mean Reward ± Std over {NUM_EPISODES} episodes — best model per (env, type)",
    fontsize=11,
    fontweight="bold",
    pad=12,
)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
