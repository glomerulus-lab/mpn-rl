"""
Plot average trial length for 'always respond' vs 'always withhold' agents
on GoNogo-v0, split by trial type (go / nogo).

This illustrates that:
  - Always Respond terminates both go and nogo trials early
    (first action=1 in the response window ends the trial immediately)
  - Always Withhold runs every trial to full length (tmax)

Usage:
    python plot_trial_lengths.py
    python plot_trial_lengths.py --n-trials 500 --out trial_lengths.png
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np

from mpn_rl.neurogym_wrapper import NeuroGymInfoWrapper

ENV_NAME = "GoNogo-v0"

AGENTS = {
    "Oracle": {"policy": "oracle", "color": "#4CAF50"},
    "Always\nWithhold": {"policy": "always_0", "color": "#9C27B0"},
    "Always\nRespond": {"policy": "always_1", "color": "#2196F3"},
    "Random": {"policy": "random", "color": "#9E9E9E"},
}


def collect_trial_lengths(policy: str, n_trials: int, seed: int):
    """Return per-trial (length, gt) pairs for the given policy."""
    env = NeuroGymInfoWrapper(ngym.make(ENV_NAME))
    env.unwrapped.rng = np.random.RandomState(seed)
    rng = np.random.RandomState(seed + 99)

    obs, info = env.reset()
    results = []
    step_count = 0

    while len(results) < n_trials:
        if policy == "always_0":
            action = 0
        elif policy == "always_1":
            action = 1
        else:
            action = int(rng.randint(0, 2))

        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1

        if info.get("new_trial", False):
            gt = int(info.get("gt", -1))
            results.append((step_count, gt))
            step_count = 0

        if terminated or truncated:
            obs, info = env.reset()
            step_count = 0

    env.close()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="trial_lengths.png")
    args = parser.parse_args()

    # Collect lengths per agent per trial type
    data = {}  # agent_name -> {0: [lengths], 1: [lengths]}
    for name, cfg in AGENTS.items():
        print(f"Collecting {args.n_trials} trials for {name.replace(chr(10), ' ')} ...")
        trials = collect_trial_lengths(cfg["policy"], args.n_trials, args.seed)
        by_type = {0: [], 1: []}
        for length, gt in trials:
            if gt in by_type:
                by_type[gt].append(length)
        data[name] = by_type
        print(
            f"  Go trials:   n={len(by_type[1])}  mean={np.mean(by_type[1]):.1f}  "
            f"std={np.std(by_type[1]):.1f}"
        )
        print(
            f"  Nogo trials: n={len(by_type[0])}  mean={np.mean(by_type[0]):.1f}  "
            f"std={np.std(by_type[0]):.1f}"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    trial_types = ["Go\n(gt=1)", "Nogo\n(gt=0)", "All trials"]
    trial_keys = [1, 0, "all"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: grouped bar chart by trial type
    ax = axes[0]
    n_agents = len(AGENTS)
    bar_w = 0.18
    x = np.arange(len(trial_types))

    for i, (name, cfg) in enumerate(AGENTS.items()):
        offset = (i - (n_agents - 1) / 2) * bar_w
        means, sems = [], []
        for k in trial_keys:
            if k == "all":
                vals = data[name][0] + data[name][1]
            else:
                vals = data[name][k]
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        bars = ax.bar(
            x + offset,
            means,
            bar_w,
            label=name.replace("\n", " "),
            color=cfg["color"],
            alpha=0.85,
            edgecolor="white",
        )
        ax.errorbar(
            x + offset, means, yerr=sems, fmt="none", color="black", capsize=3, lw=1.2
        )
        for b, m in zip(bars, means):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.15,
                f"{m:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(trial_types, fontsize=11)
    ax.set_ylabel("Mean trial length (steps)", fontsize=11)
    ax.set_title("Mean trial length by trial type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: violin distribution over all trials
    ax2 = axes[1]
    pos = 1
    tick_pos, tick_labels = [], []

    for name, cfg in AGENTS.items():
        vals = data[name][0] + data[name][1]
        vp = ax2.violinplot(
            vals, positions=[pos], widths=0.6, showmedians=True, showextrema=True
        )
        for pc in vp["bodies"]:
            pc.set_facecolor(cfg["color"])
            pc.set_alpha(0.75)
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_lw(2)
        for part in ["cbars", "cmins", "cmaxes"]:
            vp[part].set_color(cfg["color"])
        ax2.text(
            pos,
            np.mean(vals) + 0.3,
            f"{np.mean(vals):.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
        tick_pos.append(pos)
        tick_labels.append(name)
        pos += 1

    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels(tick_labels, fontsize=10)
    ax2.set_ylabel("Trial length (steps)", fontsize=11)
    ax2.set_title(
        "Trial length distribution (all trials)", fontsize=12, fontweight="bold"
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"GoNogo-v0 average trial length by agent  ({args.n_trials} trials each)",
        fontsize=13,
        fontweight="bold",
    )
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
