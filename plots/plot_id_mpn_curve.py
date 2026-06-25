"""
Training reward curve for MPN vs MPN-Frozen on IntervalDiscrimination-v0,
merged with a sample trial panel.

Outputs:
    <output>           - training curve + 1 sample trial (side by side)
    <output>_2trials   - training curve + 2 sample trials

Usage:
    python plot_id_mpn_curve.py [sweeps] [output] [seed] [env]
"""

import json
import sys

import duckdb
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np
import torch

from mpn_rl.models.actor_critic import ActorCriticNet

SWEEP = sys.argv[1] if len(sys.argv) > 1 else "ng-sweep-v1"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "id_mpn_curve.png"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 7
ENV = sys.argv[4] if len(sys.argv) > 4 else "IntervalDiscrimination-v0"
SWEEPS = [t.strip() for t in SWEEP.split(",")]
SMOOTH = 50

ENV_CONFIG = {
    "IntervalDiscrimination-v0": {"ch1": "Stim 1", "ch2": "Stim 2"},
    "DelayMatchSample-v0": {"ch1": "Sample", "ch2": "Test"},
}
_ecfg = ENV_CONFIG.get(ENV, {"ch1": "Stim 1", "ch2": "Stim 2"})

MODEL_COLORS = {
    "mpn": "#2ca02c",
    "mpn-frozen": "#d62728",
}

# ---------------------------------------------------------------------------
# Load best experiments
# ---------------------------------------------------------------------------

con = duckdb.connect()
con.execute("""
    CREATE VIEW configs AS
    SELECT * FROM read_json_auto('experiments/*/config.json', ignore_errors = true)
""")

sweeps_sql = ", ".join(f"'{t}'" for t in SWEEPS)

best_exps = con.execute(f"""
    WITH windowed AS (
        SELECT experiment_name,
            AVG(reward) OVER (
                PARTITION BY experiment_name ORDER BY frame
                ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
            ) AS reward_50,
            MAX(frame) OVER (PARTITION BY experiment_name) AS max_frame
        FROM read_ndjson('experiments/*/metrics.jsonl',
            columns={{experiment_name:'VARCHAR', frame:'INTEGER', reward:'DOUBLE'}},
            ignore_errors=true)
    ),
    run_peaks AS (
        SELECT c.experiment_name, c.sweep_name, c.model_type,
            c.num_layers, c.hidden_dim, c.learning_rate,
            MAX(w.max_frame)  AS max_frame,
            MAX(w.reward_50)  AS peak_reward
        FROM configs c JOIN windowed w ON c.experiment_name = w.experiment_name
        WHERE c.sweep_name IN ({sweeps_sql})
          AND c.env_name = '{ENV}'
          AND c.model_type IN ('mpn', 'mpn-frozen')
        GROUP BY c.experiment_name, c.sweep_name, c.model_type,
                 c.num_layers, c.hidden_dim, c.learning_rate
    ),
    deduped AS (
        SELECT * EXCLUDE(rn) FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY sweep_name, model_type, num_layers, hidden_dim, learning_rate
                    ORDER BY max_frame DESC
                ) AS rn
            FROM run_peaks
        ) WHERE rn = 1
    )
    SELECT experiment_name, model_type, sweep_name, peak_reward
    FROM deduped
    QUALIFY ROW_NUMBER() OVER (PARTITION BY model_type ORDER BY peak_reward DESC) = 1
""").fetchdf()
con.close()

for _, row in best_exps.iterrows():
    print(
        f"Best sweep for {row['model_type']}: sweep={row['sweep_name']}  peak={row['peak_reward']:.3f}  exp={row['experiment_name']}"
    )

if best_exps.empty:
    print(f"No data found for {ENV} with sweeps={SWEEPS}.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Load training curve time series
# ---------------------------------------------------------------------------


def load_training_curve(exp_name):
    episodes, rewards = [], []
    with open(f"experiments/{exp_name}/metrics.jsonl") as f:
        for line in f:
            d = json.loads(line)
            episodes.append(d["episode"])
            rewards.append(d["reward"])
    episodes = np.array(episodes)
    rewards = np.array(rewards)
    order = np.argsort(episodes)
    return episodes[order], rewards[order]


training_curves = {}
for _, row in best_exps.iterrows():
    training_curves[row["model_type"]] = load_training_curve(row["experiment_name"])

# ---------------------------------------------------------------------------
# Load trained models
# ---------------------------------------------------------------------------


def load_model(exp_name):
    cfg = json.load(open(f"experiments/{exp_name}/config.json"))
    env = ngym.make(ENV, dt=100)
    model = ActorCriticNet(
        input_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        hidden_dim=cfg["hidden_dim"],
        core_type=cfg["model_type"],
        activation=cfg.get("activation", "tanh"),
        lambda_max=cfg.get("lambda_max", 0.99),
        eta_init=cfg.get("eta_init", 0.01),
        lambda_init=cfg.get("lambda_init", 0.99),
        num_layers=cfg["num_layers"],
        mpn_bias=cfg.get("mpn_bias", True),
    )
    ckpt = torch.load(
        f"experiments/{exp_name}/checkpoints/best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    env.close()
    return model


loaded_models = {}
for _, row in best_exps.iterrows():
    loaded_models[row["model_type"]] = load_model(row["experiment_name"])

# ---------------------------------------------------------------------------
# Collect sample trials
# ---------------------------------------------------------------------------


def collect_trials(model, start_seed=7, max_steps=500):
    """Collect one trial per unique ground-truth outcome."""
    seen_outcomes = {}
    model.eval()
    with torch.no_grad():
        s = start_seed
        while s < start_seed + 500:
            env = ngym.make(ENV, dt=100)
            env.unwrapped.rng = np.random.RandomState(s)
            obs, _ = env.reset()
            h = None
            trial_obs, trial_gt, trial_rew = [obs], [0], [0.0]
            for _ in range(max_steps):
                x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                action_probs, _, h = model(x, h)
                action = int(torch.argmax(action_probs, dim=-1).item())
                obs, reward, terminated, truncated, info = env.step(action)
                trial_obs.append(obs)
                trial_gt.append(int(info.get("gt", 0)))
                trial_rew.append(float(reward))
                if terminated or truncated or info.get("new_trial"):
                    break
            env.close()
            outcome = max(trial_gt)
            if outcome not in seen_outcomes:
                seen_outcomes[outcome] = (np.array(trial_obs), trial_gt, trial_rew)
            if len(seen_outcomes) >= 2:
                break
            s += 1
    model.train()
    return list(seen_outcomes.values())


def get_trial_actions(trial_list):
    result = []
    for t_obs, _, _ in trial_list:
        actions = {}
        for mt, model in loaded_models.items():
            probs_list = []
            with torch.no_grad():
                h = None
                for obs in t_obs:
                    x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    action_probs, _, h = model(x, h)
                    probs_list.append(action_probs.numpy()[0])
            actions[mt] = np.argmax(np.array(probs_list), axis=1).astype(float)
        result.append(actions)
    return result


mpn_model = loaded_models.get("mpn")
trials = collect_trials(mpn_model, start_seed=SEED)
trial_actions = get_trial_actions(trials)

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

STIM_COLOR = "#7b2d8b"

CHANNEL_DEFS = [
    ("MPN-frozen", MODEL_COLORS["mpn-frozen"]),
    ("MPN", MODEL_COLORS["mpn"]),
    ("Ground Truth", "#444444"),
    (_ecfg["ch1"], STIM_COLOR),
    (_ecfg["ch2"], STIM_COLOR),
    ("Fixation", "#888888"),
]

OFFSET = 3.5


def plot_training_curve(ax):
    for mt, (frames, rewards) in training_curves.items():
        color = MODEL_COLORS[mt]
        label = "MPN" if mt == "mpn" else "MPN-frozen"
        if len(rewards) >= SMOOTH:
            smoothed = np.convolve(rewards, np.ones(SMOOTH) / SMOOTH, mode="valid")
            smooth_frames = frames[SMOOTH - 1 :]
            ax.plot(smooth_frames, smoothed, color=color, linewidth=2, label=label)
        else:
            ax.plot(frames, rewards, color=color, linewidth=2, label=label)
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Reward", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def plot_trial_panel(ax, t_obs, t_gt, t_rew, actions, show_ylabels=True):
    t_ms = np.arange(len(t_obs))

    channels = [
        actions.get("mpn-frozen", np.zeros(len(t_obs))),
        actions.get("mpn", np.zeros(len(t_obs))),
        np.full(len(t_obs), max(t_gt), dtype=float),
        t_obs[:, 1],
        t_obs[:, 2],
        t_obs[:, 0],
    ]

    for ci, (signal, (_, color)) in enumerate(zip(channels, CHANNEL_DEFS)):
        offset = ci * OFFSET
        ax.step(t_ms, signal + offset, where="post", color=color, linewidth=1.5)
        ref_vals = [0.0, 1.0, 2.0] if ci < 3 else [0.0, 1.0]
        for val in ref_vals:
            ax.hlines(
                offset + val,
                t_ms[0],
                t_ms[-1],
                color="#222222",
                linewidth=0.8,
                linestyle=(0, (8, 3)),
                alpha=0.7,
                zorder=1,
            )

    mid_positions = [
        ci * OFFSET + (1.0 if ci < 3 else 0.5) for ci in range(len(CHANNEL_DEFS))
    ]

    if show_ylabels:
        ax.set_yticks(mid_positions)
        ax.set_yticklabels(
            [lbl for lbl, _ in CHANNEL_DEFS], fontsize=14, fontweight="bold"
        )
        for tick, (_, color) in zip(ax.get_yticklabels(), CHANNEL_DEFS):
            tick.set_color(color)
        ax.tick_params(axis="y", length=0, pad=5)
    else:
        ax.set_yticks([])

    fix_off = np.where(t_obs[:, 0] == 0)[0]
    if len(fix_off):
        dec_idx = fix_off[0]
        y_min = -0.3
        y_max = (len(CHANNEL_DEFS) - 1) * OFFSET + 2.5
        fix1_y = (len(CHANNEL_DEFS) - 1) * OFFSET + 1.0
        span_ymax = (fix1_y - y_min) / (y_max - y_min)
        ax.axvspan(
            t_ms[dec_idx],
            t_ms[dec_idx] + 1,
            ymin=0,
            ymax=span_ymax,
            color="#ff7f0e",
            alpha=0.25,
            zorder=0,
        )

    trial_len = len(t_obs)
    ax.set_xlim(0, trial_len - 1)
    ax.set_xticks(np.arange(0, trial_len + 1))
    ax.set_xticklabels(
        [str(x) if x % 5 == 0 else "" for x in np.arange(0, trial_len + 1)]
    )
    ax.set_ylim(-0.3, (len(CHANNEL_DEFS) - 1) * OFFSET + 2.5)
    ax.tick_params(labelsize=13)
    ax.grid(axis="x", alpha=0.2, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Frame", fontsize=16)


# ---------------------------------------------------------------------------
# Figure 1: training curve + 1 trial
# ---------------------------------------------------------------------------


def make_figure():
    fig = plt.figure(figsize=(26, 7.5))
    gs_outer = gridspec.GridSpec(1, 2, width_ratios=[1.5, 2.4], wspace=0.3, figure=fig)
    gs_trials = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs_outer[1], wspace=0.12
    )
    ax_curve = fig.add_subplot(gs_outer[0])
    ax_t1 = fig.add_subplot(gs_trials[0])
    ax_t2 = fig.add_subplot(gs_trials[1])
    return fig, ax_curve, ax_t1, ax_t2


def add_panel_label(ax, label):
    ax.text(
        -0.08,
        1.05,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="right",
    )


fig1, ax_curve1, ax_t1, ax_t2 = make_figure()
plot_training_curve(ax_curve1)
plot_trial_panel(ax_t1, *trials[0], trial_actions[0], show_ylabels=True)
plot_trial_panel(ax_t2, *trials[1], trial_actions[1], show_ylabels=False)

add_panel_label(ax_curve1, "(a)")
add_panel_label(ax_t1, "(b)")

fig1.savefig(OUTPUT, dpi=600, bbox_inches="tight")
print(f"Saved → {OUTPUT}")

# ---------------------------------------------------------------------------
# Figure 2: training curve + 2 trials (duplicate for legacy output name)
# ---------------------------------------------------------------------------

out_2t = OUTPUT.replace(".png", "_2trials.png")
fig2, ax_curve2, ax_t3, ax_t4 = make_figure()
plot_training_curve(ax_curve2)
plot_trial_panel(ax_t3, *trials[0], trial_actions[0], show_ylabels=True)
plot_trial_panel(ax_t4, *trials[1], trial_actions[1], show_ylabels=False)

fig2.savefig(out_2t, dpi=600, bbox_inches="tight")
print(f"Saved → {out_2t}")
