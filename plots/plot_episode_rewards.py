"""Plot GoNogo and IntervalDiscrimination learning curves by episode number."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mpn_rl.runs import load_runs

COLORS = {
    "rnn": "#1f77b4",
    "lstm": "#ff7f0e",
    "mpn": "#2ca02c",
    "mpn_frozen": "#d62728",
}
SMOOTH = 50

TASKS = {
    "GoNogo": "gonogo",
    "IntervalDiscrimination": "intdisc",
}


def smooth(x, w):
    if len(x) < w:
        return np.arange(len(x)), x
    s = np.convolve(x, np.ones(w) / w, mode="valid")
    offset = len(x) - len(s)
    return np.arange(offset, len(x)), s


def load_jsonl(path):
    rewards = []
    with open(path) as fh:
        for line in fh:
            rewards.append(json.loads(line)["reward"])
    return np.array(rewards)


runs = load_runs()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (task_name, task_key) in zip(axes, TASKS.items()):
    prefix = f"a2c_run2-{task_key}-"
    matched = runs[runs["experiment_name"].str.startswith(prefix)]
    matched = matched.sort_values("experiment_name")
    plotted = set()
    for _, run in matched.iterrows():
        d = Path(run["path"])
        mf = d / "metrics.jsonl"
        if not mf.exists() or mf.stat().st_size == 0:
            continue
        model = run["experiment_name"].split(prefix)[1].rsplit("-", 1)[0]
        rewards = load_jsonl(mf)
        episodes = np.arange(len(rewards))
        color = COLORS.get(model, "grey")
        label = model if model not in plotted else None
        ep_s, r_s = smooth(rewards, SMOOTH)
        ax.plot(episodes, rewards, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(ep_s, r_s, label=label, color=color, linewidth=2)
        plotted.add(model)

    ax.set_title(task_name, fontsize=13)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.4)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("plot_episode_rewards.png", dpi=150, bbox_inches="tight")
print("Saved plot_episode_rewards.png")
