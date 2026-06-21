"""Greedy evaluation of a trained ActorCriticNet."""

import numpy as np
import torch


def _evaluate_actorcritic(model, env_factory, num_episodes, max_steps, seed, device):
    """Evaluate ActorCriticNet greedily on a fresh env from env_factory.

    Returns (mean_reward, std_reward, per_episode_rewards).
    """
    model.eval()
    rewards = []
    with torch.no_grad():
        for ep in range(num_episodes):
            env = env_factory()
            if seed is not None:
                env.unwrapped.rng = np.random.RandomState(seed + ep)
            obs, _ = env.reset()
            h = None
            ep_reward = 0.0
            for _ in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                policy_dist, _, h = model(obs_t, h)
                action = int(policy_dist.argmax(-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            env.close()
    model.train()
    return float(np.mean(rewards)), float(np.std(rewards)), rewards
