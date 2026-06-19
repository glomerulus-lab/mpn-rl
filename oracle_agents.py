"""
Oracle agents for NeuroGym environments.

Each oracle reads env.gt_now (the ground-truth action supplied by the
neurogym task) at every step.  This gives optimal behaviour for all 15
paper environments:

  - Choice tasks (GoNogo, PDM, DMS, etc.): gt_now = correct action during the
    decision window, 0 (fixate) elsewhere.
  - Timing tasks (ReadySetGo, MotorTiming, OneTwoThreeGo): gt_now = 1 at the
    single step that maximises reward, 0 at every other step.
  - Multi-decision tasks (DualDelayMatchSample, DMSD1D): gt_now is correct for
    each test window.

Usage
-----
    from oracle_agents import evaluate_oracle

    oracle_reward = evaluate_oracle("GoNogo-v0", n_episodes=100,
                                    max_steps=10_000)
    pct = eval_reward / oracle_reward * 100
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


class GTOracle:
    """Universal oracle that follows env.gt_now at every step.

    Works for all neurogym TrialEnv environments: the environment itself
    exposes the ground-truth action via ``env.gt_now``, which is 0 (fixate /
    hold) outside of the response window and the correct discrete action
    inside it.
    """

    def reset(self) -> None:
        pass  # stateless

    def act(self, obs: np.ndarray, unwrapped_env) -> int:
        """Return the ground-truth action for the current step.

        Args:
            obs: Current observation (unused — oracle peeks at the env).
            unwrapped_env: The unwrapped neurogym TrialEnv instance,
                giving access to ``gt_now``.

        Returns:
            Integer action.
        """
        return int(unwrapped_env.gt_now)


def evaluate_oracle(
    env_name: str,
    n_episodes: int = 200,
    max_steps: int = 10_000,
    seed: Optional[int] = 0,
    env_factory=None,
) -> float:
    """Run the GTOracle and return the mean per-episode reward.

    Args:
        env_name: NeuroGym environment ID (e.g. ``"GoNogo-v0"``). Used only
            when *env_factory* is None.
        n_episodes: Number of full episodes to average over.
        max_steps: Hard step cap per episode (mirrors training
            ``max_episode_steps``).
        seed: Optional RNG seed for the environment.
        env_factory: Optional callable that returns a fresh neurogym env with
            the correct reward configuration. When provided, *env_name* is
            ignored. Pass this to keep oracle rewards consistent with the
            training environment.

    Returns:
        Mean total reward per episode achieved by the perfect oracle.
    """
    import neurogym as ngym  # imported lazily so the module is importable

    # even without neurogym installed

    if env_factory is not None:
        env = env_factory()
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = ngym.make(env_name)

    oracle = GTOracle()
    episode_rewards: list[float] = []

    for ep_idx in range(n_episodes):
        # TrialEnv uses self.rng (np.random.RandomState) for all trial
        # randomness; reset(seed=...) only updates gymnasium's np_random and
        # does NOT reseed self.rng.  Reseed it directly for reproducibility.
        ep_seed = (seed + ep_idx) if seed is not None else None
        if ep_seed is not None:
            env.unwrapped.rng = np.random.RandomState(ep_seed)
        obs, _ = env.reset()
        oracle.reset()
        ep_reward = 0.0
        for _ in range(max_steps):
            action = oracle.act(obs, env.unwrapped)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break
        episode_rewards.append(ep_reward)

    env.close()
    return float(np.mean(episode_rewards))


def get_oracle_reward(
    env_name: str,
    n_episodes: int = 200,
    max_steps: int = 10_000,
    seed: int = 0,
    env_factory=None,
) -> float:
    """Evaluate the oracle and return the mean reward.

    Pass *env_factory* to ensure the oracle uses the same reward configuration
    as the training environment (e.g. after reward patching). When omitted,
    a fresh environment is created from *env_name* with default rewards.
    """
    return evaluate_oracle(
        env_name,
        n_episodes=n_episodes,
        max_steps=max_steps,
        seed=seed,
        env_factory=env_factory,
    )


if __name__ == "__main__":
    """Quick sanity check across all paper environments."""
    PAPER_ENVS = [
        "ContextDecisionMaking-v0",
        "DelayComparison-v0",
        "DelayMatchSample-v0",
        "DelayMatchSampleDistractor1D-v0",
        "DelayPairedAssociation-v0",
        "DualDelayMatchSample-v0",
        "GoNogo-v0",
        "IntervalDiscrimination-v0",
        "MotorTiming-v0",
        "MultiSensoryIntegration-v0",
        "OneTwoThreeGo-v0",
        "PerceptualDecisionMaking-v0",
        "PerceptualDecisionMakingDelayResponse-v0",
        "ProbabilisticReasoning-v0",
        "ReadySetGo-v0",
    ]

    print(f"{'Environment':<45} {'Oracle Reward':>14}")
    print("-" * 62)
    for env_name in PAPER_ENVS:
        r = evaluate_oracle(env_name, n_episodes=200, seed=42)
        print(f"{env_name:<45} {r:>14.3f}")
