"""Greedy evaluation of trained agents."""

import numpy as np
import torch
import tqdm

from mpn_rl.models.supervised import SupervisedNet
from mpn_rl.supervised_data import MaskedSequenceSampler


def evaluate_actorcritic(
    model, env_factory, num_episodes, max_steps, seed, device, progress=False
):
    """Evaluate ActorCriticNet greedily on a fresh env from env_factory.

    Returns (per-episode rewards, per-episode lengths). progress=True shows a
    tqdm bar.
    """
    model.eval()
    rewards = []
    lengths = []
    bar = tqdm.tqdm(
        range(num_episodes), desc="Evaluating", unit="episode", disable=not progress
    )
    with torch.no_grad():
        for ep in bar:
            env = env_factory()
            if seed is not None:
                env.unwrapped.rng = np.random.RandomState(seed + ep)
            obs, _ = env.reset()
            state = None
            ep_reward = 0.0
            ep_length = 0
            for _ in range(max_steps):
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                policy_dist, _, state = model(obs_t, state)
                action = int(policy_dist.argmax(-1).item())
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_reward += reward
                ep_length += 1
                if terminated or truncated:
                    break
            rewards.append(ep_reward)
            lengths.append(ep_length)
            env.close()
            bar.set_postfix(reward=f"{ep_reward:.2f}")
    model.train()
    return rewards, lengths


def evaluate_supervised(
    model: SupervisedNet,
    sampler: MaskedSequenceSampler,
    num_sequences: int,
    device: str,
) -> float:
    """Masked accuracy over num_sequences freshly sampled sequences."""
    model.eval()
    correct = total = 0
    seen = 0
    with torch.no_grad():
        while seen < num_sequences:
            inputs_np, targets_np, mask_np = sampler.sample()
            inputs = torch.as_tensor(inputs_np, dtype=torch.float32, device=device)
            targets = torch.as_tensor(targets_np, dtype=torch.long, device=device)
            mask = torch.as_tensor(mask_np, device=device)
            pred = model(inputs).argmax(-1)
            correct += int((pred == targets)[mask].sum())
            total += int(mask.sum())
            seen += inputs.shape[0]
    model.train()
    return correct / total if total else 0.0
