"""
Display reward position within sampled trials for each NeuroGym environment.

Simulates what the replay buffer sees: rolls out a random policy, splits on
new_trial boundaries, and prints a clear table showing where reward falls
within each trial (relative to trial start and end).

Usage:
    python check_reward_timing.py
    python check_reward_timing.py --n-trials 8 --seed 0
"""

import argparse
import warnings

warnings.filterwarnings("ignore")

import neurogym as ngym
import numpy as np

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
    "DelayPairedAssociation-v0": "DelayPairedAssoc",
    "IntervalDiscrimination-v0": "IntervalDisc",
    "MultiSensoryIntegration-v0": "MultiSensory",
    "PerceptualDecisionMaking-v0": "PDM",
    "PerceptualDecisionMakingDelayResponse-v0": "PDM-DelayResp",
    "ProbabilisticReasoning-v0": "ProbReasoning",
}


def collect_trials(env_name: str, n_trials: int, seed: int):
    """Roll out a random policy and return complete trials split on new_trial."""
    env = ngym.make(env_name)
    env.unwrapped.rng = np.random.RandomState(seed)
    env.reset()

    trials = []
    current = []

    for _ in range(n_trials * 200):  # generous step budget
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        current.append(reward)

        if info.get("new_trial", False):
            if current:
                trials.append(current)
                current = []
        if terminated or truncated:
            if current:
                trials.append(current)
            env.reset()
            current = []

        if len(trials) >= n_trials:
            break

    return trials[:n_trials]


def reward_bar(rewards, width=40):
    """ASCII bar showing reward magnitude at each step (scaled to width)."""
    n = len(rewards)
    bar = []
    for r in rewards:
        if r > 0:
            bar.append("+")
        elif r < 0:
            bar.append("-")
        else:
            bar.append(".")
    # Pad/truncate to width
    bar_str = "".join(bar)
    if n <= width:
        bar_str = bar_str.ljust(width)
    else:
        # Compress: sample at regular intervals
        indices = np.linspace(0, n - 1, width, dtype=int)
        bar_str = "".join(bar[i] for i in indices)
    return bar_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for env_name in ENVS:
        short = ENV_SHORT.get(env_name, env_name)
        print(f"\n{'='*70}")
        print(f"  {short}  ({env_name})")
        print(f"{'='*70}")

        try:
            trials = collect_trials(env_name, args.n_trials, args.seed)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not trials:
            print("  No trials collected.")
            continue

        # Summary stats
        lengths = [len(t) for t in trials]
        any_rew = [any(r != 0 for r in t) for t in trials]
        rew_steps = []
        for t in trials:
            for i, r in enumerate(t):
                if r != 0:
                    rew_steps.append(i)

        last_rew_frac = []
        for t in trials:
            nonzero = [i for i, r in enumerate(t) if r != 0]
            if nonzero:
                last_rew_frac.append(nonzero[-1] / (len(t) - 1))

        print(
            f"  Avg trial length : {np.mean(lengths):.1f} steps  "
            f"(min {min(lengths)}, max {max(lengths)})"
        )
        print(f"  Trials with reward: {sum(any_rew)}/{len(trials)}")
        if last_rew_frac:
            print(
                f"  Last reward at   : {np.mean(last_rew_frac)*100:.0f}% through trial "
                f"(1.0 = final step)"
            )
        print()

        # Per-trial display
        # Header
        print(
            f"  {'Trial':<6} {'Len':>4}  {'Reward steps':<28}  {'Bar (. = 0, + = pos, - = neg)'}"
        )
        print(f"  {'-'*6} {'-'*4}  {'-'*28}  {'-'*40}")
        for idx, trial in enumerate(trials):
            nonzero = [(i, trial[i]) for i in range(len(trial)) if trial[i] != 0]
            rew_str = (
                "  ".join(f"t={i}({r:+.1f})" for i, r in nonzero) if nonzero else "none"
            )
            bar = reward_bar(trial)
            print(f"  {idx:<6} {len(trial):>4}  {rew_str:<28}  {bar}")

        print()


if __name__ == "__main__":
    main()
