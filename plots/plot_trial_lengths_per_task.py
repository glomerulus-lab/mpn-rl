"""
Average trial length per environment, measured using the oracle (GT) agent.

Usage:
    python plot_trial_lengths_per_task.py [output]
"""

import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np

from mpn_rl.neurogym_wrapper import NeuroGymInfoWrapper

OUTPUT = sys.argv[1] if len(sys.argv) > 1 else "trial_lengths_per_task.png"
N_TRIALS = 500
SEED = 42

ENVS = [
    "GoNogo-v0",
    "ContextDecisionMaking-v0",
    "DelayComparison-v0",
    "DelayMatchSample-v0",
    "DelayMatchSampleDistractor1D-v0",
    "DelayPairedAssociation-v0",
    "IntervalDiscrimination-v0",
    "MultiSensoryIntegration-v0",
    "PerceptualDecisionMaking-v0",
    "PerceptualDecisionMakingDelayResponse-v0",
    "ProbabilisticReasoning-v0",
]

ENV_SHORT = {
    "GoNogo-v0": "GoNogo",
    "ContextDecisionMaking-v0": "ContextDM",
    "DelayComparison-v0": "DelayComp",
    "DelayMatchSample-v0": "DMS",
    "DelayMatchSampleDistractor1D-v0": "DMSD1D",
    "DelayPairedAssociation-v0": "DelayPA",
    "IntervalDiscrimination-v0": "IntervalDisc",
    "MultiSensoryIntegration-v0": "MultiSens",
    "PerceptualDecisionMaking-v0": "PDM",
    "PerceptualDecisionMakingDelayResponse-v0": "PDM-DelayResp",
    "ProbabilisticReasoning-v0": "ProbReason",
}


def collect_oracle_lengths(env_name: str, n_trials: int, seed: int) -> np.ndarray:
    env = NeuroGymInfoWrapper(ngym.make(env_name))
    env.unwrapped.rng = np.random.RandomState(seed)
    obs, info = env.reset()
    lengths, step = [], 0
    while len(lengths) < n_trials:
        action = int(env.unwrapped.gt_now)
        obs, _, term, trunc, info = env.step(action)
        step += 1
        if info.get("new_trial", False):
            lengths.append(step)
            step = 0
        if term or trunc:
            obs, info = env.reset()
            step = 0
    env.close()
    return np.array(lengths)


means, stds, labels = [], [], []
for env_name in ENVS:
    print(f"  {ENV_SHORT[env_name]:<18}", end="", flush=True)
    lengths = collect_oracle_lengths(env_name, N_TRIALS, SEED)
    m, s = lengths.mean(), lengths.std()
    means.append(m)
    stds.append(s)
    labels.append(ENV_SHORT[env_name])
    print(f"mean={m:.1f}  std={s:.1f}")

# Sort by mean length
order = np.argsort(means)
means = [means[i] for i in order]
stds = [stds[i] for i in order]
labels = [labels[i] for i in order]

fig, ax = plt.subplots(figsize=(7, 5))

y = np.arange(len(ENVS))
bars = ax.barh(
    y,
    means,
    xerr=stds,
    color="#4C72B0",
    alpha=0.85,
    error_kw=dict(ecolor="#222", capsize=3, lw=1.2),
    edgecolor="white",
)

for yi, (m, s) in enumerate(zip(means, stds)):
    ax.text(m + s + 0.4, yi, f"{m:.0f}", va="center", ha="left", fontsize=9)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel("Mean trial length (steps)", fontsize=11)
ax.set_title(
    "Average trial length per environment\n(oracle agent, 500 trials)", fontsize=11
)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"\nSaved → {OUTPUT}")
