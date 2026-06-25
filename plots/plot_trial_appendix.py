"""
Sample trial plots for appendix figures.
Plots 3 trials side by side for the best MPN on a given environment.

Usage:
    python plot_trial_appendix.py <env_name> <output> [seed]

    env_name  - neurogym environment id (e.g. PerceptualDecisionMaking-v0)
    output    - output file path (default: trial_appendix.png)
    seed      - starting seed for trial collection (default: 0)
"""

import json
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np
import torch

from mpn_rl.experiment import find_experiment_files, load_experiments
from mpn_rl.models.actor_critic import ActorCriticNet

ENV = sys.argv[1] if len(sys.argv) > 1 else "PerceptualDecisionMaking-v0"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "trial_appendix.png"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
_sweeps_arg = sys.argv[4] if len(sys.argv) > 4 else "ng-sweep-v1"
SWEEPS = [s.strip() for s in _sweeps_arg.split(",")]

MODEL_COLOR = "#2ca02c"
GT_COLOR = "#444444"
FIX_COLOR = "#888888"
STIM_COLOR = "#7b2d8b"


# Per-environment channel extraction: returns list of (label, color, signal_array)
# All signals should be in [0, 1].
def extract_channels(t_obs, actions):
    fix = t_obs[:, 0]
    if t_obs.shape[1] == 3:
        # PDM / PDMDR: [fixation, stim1, stim2]
        return [
            ("MPN", MODEL_COLOR, actions),
            ("Stim 1", STIM_COLOR, t_obs[:, 1]),
            ("Stim 2", STIM_COLOR, t_obs[:, 2]),
            ("Fixation", FIX_COLOR, fix),
        ]
    else:
        # ProbabilisticReasoning: [fixation, left_stims(1:21), right_stims(21:41)]
        left = t_obs[:, 1:21].sum(axis=1).clip(0, 1)
        right = t_obs[:, 21:41].sum(axis=1).clip(0, 1)
        return [
            ("MPN", MODEL_COLOR, actions),
            ("Left stim", STIM_COLOR, left),
            ("Right stim", STIM_COLOR, right),
            ("Fixation", FIX_COLOR, fix),
        ]


# ---------------------------------------------------------------------------
# Find best experiment
# ---------------------------------------------------------------------------

con = duckdb.connect()
metrics_list = (
    "["
    + ", ".join(f"'{p}'" for p in find_experiment_files("metrics.jsonl", None))
    + "]"
)
con.register("configs", load_experiments())
row = con.execute(
    f"""
    WITH windowed AS (
        SELECT experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name ORDER BY frame
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS reward_50
        FROM read_ndjson({metrics_list},
            columns={{experiment_name:'VARCHAR',frame:'INTEGER',reward:'DOUBLE'}},
            ignore_errors=true)
    )
    SELECT c.experiment_name, c.path, MAX(w.reward_50) AS peak
    FROM configs c
    JOIN windowed w ON c.experiment_name = w.experiment_name
    WHERE c.sweep_name IN ({", ".join("?" for _ in SWEEPS)})
      AND c.env_name = ?
      AND c.model_type = 'mpn'
    GROUP BY c.experiment_name, c.path
    ORDER BY peak DESC LIMIT 1
""",
    [*SWEEPS, ENV],
).fetchone()
con.close()

if row is None:
    print(f"No MPN experiment found for {ENV}")
    sys.exit(1)

exp_name, run_path, peak = row
print(f"Best MPN for {ENV}: {exp_name}  peak={peak:.3f}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

run_dir = Path(run_path)
cfg = json.load(open(run_dir / "config.json"))
env0 = ngym.make(ENV, dt=100)
model = ActorCriticNet(
    input_dim=env0.observation_space.shape[0],
    action_dim=env0.action_space.n,
    hidden_dim=cfg["hidden_dim"],
    model_type=cfg["model_type"],
    activation=cfg.get("activation", "tanh"),
    lambda_max=cfg.get("lambda_max", 0.99),
    eta_init=cfg.get("eta_init", 0.01),
    lambda_init=cfg.get("lambda_init", 0.99),
    num_layers=cfg["num_layers"],
    mpn_bias=cfg.get("mpn_bias", True),
)
ckpt = torch.load(
    run_dir / "checkpoints" / "best_model.pt",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
env0.close()

# ---------------------------------------------------------------------------
# Collect 3 trials
# ---------------------------------------------------------------------------

DECISION_PAD = 2  # minimum decision-period frames to show


def collect_trials(n=3, start_seed=0, max_steps=500):
    trials = []
    s = start_seed
    model.eval()
    with torch.no_grad():
        while len(trials) < n and s < start_seed + 2000:
            env = ngym.make(ENV, dt=100)
            env.unwrapped.rng = np.random.RandomState(s)
            obs, _ = env.reset()
            h = None
            t_obs, t_gt, t_rew, t_act = [obs.copy()], [0], [0.0], [0]
            for _ in range(max_steps):
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_probs, _, h = model(x, h)
                action = int(torch.argmax(action_probs, dim=-1).item())
                obs_new, reward, terminated, truncated, info = env.step(action)
                gt = int(info.get("gt", 0))
                is_done = terminated or truncated or bool(info.get("new_trial"))
                if is_done:
                    # Append the decision-period obs returned by the env
                    t_obs.append(obs_new.copy())
                    t_gt.append(gt)
                    t_rew.append(float(reward))
                    t_act.append(action)
                    break
                obs = obs_new
                t_obs.append(obs.copy())
                t_gt.append(gt)
                t_rew.append(float(reward))
                t_act.append(action)
            env.close()

            # Skip trials where the model responded incorrectly
            t_gt_arr = np.array(t_gt)
            dec_frames = np.where(t_gt_arr > 0)[0]
            if len(dec_frames) == 0:
                s += 1
                continue
            first_dec = int(dec_frames[0])
            if int(t_act[first_dec]) != int(t_gt_arr[first_dec]):
                s += 1
                continue

            # Pad decision period to at least DECISION_PAD visible frames
            if len(dec_frames) < DECISION_PAD:
                first_dec = int(dec_frames[0])
                pad_obs = np.array(t_obs)[first_dec].copy()
                pad_gt = int(t_gt_arr[first_dec])
                pad_act = t_act[first_dec]
                for _ in range(DECISION_PAD - len(dec_frames)):
                    t_obs.append(pad_obs.copy())
                    t_gt.append(pad_gt)
                    t_rew.append(0.0)
                    t_act.append(pad_act)

            trials.append((np.array(t_obs), t_gt, t_rew, np.array(t_act, dtype=float)))
            s += 1
    return trials[:n]


trials = collect_trials(n=3, start_seed=SEED)

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

OFFSET = 3.5
N_TRIALS = len(trials)

fig, axes = plt.subplots(1, N_TRIALS, figsize=(6 * N_TRIALS, 7), sharey=False)
if N_TRIALS == 1:
    axes = [axes]

for col, (t_obs, t_gt, t_rew, t_act) in enumerate(trials):
    ax = axes[col]
    channels = extract_channels(t_obs, t_act)

    # Ground truth inserted after MPN action row
    gt_signal = np.full(len(t_obs), float(max(t_gt)))
    all_rows = (
        [
            (channels[0][0], channels[0][1], channels[0][2]),  # MPN action
            ("Ground Truth", GT_COLOR, gt_signal),
        ]
        + [(lbl, col_, sig) for lbl, col_, sig in channels[1:]]
    )

    for ci, (lbl, color, signal) in enumerate(all_rows):
        off = ci * OFFSET
        ax.step(
            np.arange(len(signal)),
            signal + off,
            where="post",
            color=color,
            linewidth=1.5,
        )
        ref_vals = [0.0, 1.0, 2.0] if ci < 2 else [0.0, 1.0]
        for val in ref_vals:
            ax.hlines(
                off + val,
                0,
                len(signal) - 1,
                color="#222222",
                linewidth=0.8,
                linestyle=(0, (8, 3)),
                alpha=0.7,
                zorder=1,
            )

    # Decision period shading — shade all frames where gt > 0
    dec_frames = np.where(np.array(t_gt) > 0)[0]
    if len(dec_frames):
        dec_start = dec_frames[0]
        dec_end = dec_frames[-1] + 1
        ax.axvspan(
            dec_start,
            dec_end - 1,
            ymin=0,
            ymax=1,
            color="#ff7f0e",
            alpha=0.25,
            zorder=0,
        )

    trial_len = len(t_obs)
    ax.set_xlim(0, trial_len)
    ax.set_xticks(np.arange(0, trial_len + 1))
    ax.set_xticklabels(
        [str(x) if x % 5 == 0 else "" for x in np.arange(0, trial_len + 1)]
    )
    ax.set_ylim(-0.3, (len(all_rows) - 1) * OFFSET + 2.5)
    ax.set_xlabel("Frame", fontsize=13)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)

    if col == 0:
        mid_positions = [
            ci * OFFSET + (1.0 if ci < 2 else 0.5) for ci in range(len(all_rows))
        ]
        ax.set_yticks(mid_positions)
        ax.set_yticklabels(
            [lbl for lbl, _, _ in all_rows], fontsize=11, fontweight="bold"
        )
        for tick, (_, color, _) in zip(ax.get_yticklabels(), all_rows):
            tick.set_color(color)
        ax.tick_params(axis="y", length=0, pad=5)
    else:
        ax.set_yticks([])

    ax.set_title(f"Trial {col + 1}", fontsize=11)

fig.suptitle(ENV.replace("-v0", ""), fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved → {OUTPUT}")
