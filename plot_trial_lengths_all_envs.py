"""
Compare average trial length for always-withhold vs always-respond
across all 11 NeuroGym environments.

Shows which environments have early-termination on respond (action=1).

Usage:
    python plot_trial_lengths_all_envs.py
    python plot_trial_lengths_all_envs.py --n-trials 500 --out trial_lengths_all.png
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import neurogym as ngym
import numpy as np

from neurogym_wrapper import NeuroGymInfoWrapper

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

POLICIES = {
    "Always Withhold": {"policy": "always_0", "color": "#9C27B0"},
    "Always Respond": {"policy": "always_1", "color": "#2196F3"},
}


def collect_lengths(env_name: str, policy: str, n_trials: int, seed: int):
    env = NeuroGymInfoWrapper(ngym.make(env_name))
    env.unwrapped.rng = np.random.RandomState(seed)
    obs, info = env.reset()
    lengths, step = [], 0
    while len(lengths) < n_trials:
        action = 0 if policy == "always_0" else 1
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        if info.get("new_trial", False):
            lengths.append(step)
            step = 0
        if term or trunc:
            obs, info = env.reset()
            step = 0
    env.close()
    return np.array(lengths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="trial_lengths_all_envs.png")
    args = parser.parse_args()

    # Collect data
    results = {}  # env_name -> {policy_name: lengths_array}
    for env_name in ENVS:
        print(f"{ENV_SHORT[env_name]:<16}", end="  ", flush=True)
        results[env_name] = {}
        for pol_name, cfg in POLICIES.items():
            lengths = collect_lengths(env_name, cfg["policy"], args.n_trials, args.seed)
            results[env_name][pol_name] = lengths
            print(f"{pol_name}: {np.mean(lengths):.1f}", end="  ", flush=True)
        diff = np.mean(results[env_name]["Always Withhold"]) - np.mean(
            results[env_name]["Always Respond"]
        )
        print(f"diff={diff:.1f}  {'← EARLY TERM' if diff > 0.5 else ''}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_envs = len(ENVS)
    bar_w = 0.35
    x = np.arange(n_envs)
    offsets = [-bar_w / 2, bar_w / 2]

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)

    for (pol_name, cfg), offset in zip(POLICIES.items(), offsets):
        means = [np.mean(results[e][pol_name]) for e in ENVS]
        sems = [np.std(results[e][pol_name]) / np.sqrt(args.n_trials) for e in ENVS]
        ax.bar(
            x + offset,
            means,
            bar_w,
            label=pol_name,
            color=cfg["color"],
            alpha=0.85,
            edgecolor="white",
        )
        ax.errorbar(
            x + offset, means, yerr=sems, fmt="none", color="black", capsize=3, lw=1.2
        )

    # Shade environments with early termination and annotate diff
    for i, env_name in enumerate(ENVS):
        m0 = np.mean(results[env_name]["Always Withhold"])
        m1 = np.mean(results[env_name]["Always Respond"])
        diff = m0 - m1
        if diff > 0.5:
            ax.axvspan(i - 0.5, i + 0.5, color="#FFEB3B", alpha=0.18, zorder=0)
            ax.text(
                i,
                max(m0, m1) + 1.0,
                f"−{diff:.0f} steps",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#D32F2F",
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [ENV_SHORT[e] for e in ENVS], rotation=30, ha="right", fontsize=10
    )
    ax.set_ylabel("Mean trial length (steps)", fontsize=12)
    ax.set_title(
        f"Early-termination on respond: average trial length across NeuroGym environments\n"
        f"({args.n_trials} trials each · yellow = early-termination environment)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"\nSaved → {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
