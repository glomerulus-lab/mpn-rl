import random
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import tqdm
import tyro

from envs import _create_env_from_config, _load_model_from_config
from model_utils import ExperimentManager, get_device


@dataclass
class EvalConfig:
    """Evaluate a trained agent."""

    experiment_name: str
    experiments_dir: Annotated[
        Path, tyro.conf.arg(help="Root dir the experiment was logged under")
    ] = Path("experiments")
    num_eval_episodes: int = 10
    checkpoint: str | None = None
    max_episode_steps: int = 500
    device: str = "cpu"


def evaluate(args: EvalConfig):
    """Evaluate trained agent."""
    print("=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    exp_manager = ExperimentManager(args.experiments_dir, args.experiment_name)
    config = exp_manager.load_config()

    device = get_device(args.device)
    print(f"Model type: {config.get('model_type', 'lstm').upper()}\n")

    model = _load_model_from_config(config, device)
    checkpoint_name = args.checkpoint if args.checkpoint else "best_model.pt"
    exp_manager.load_model(model, checkpoint_name=checkpoint_name, device=str(device))
    model.eval()

    env = _create_env_from_config(config)
    print(f"Loaded checkpoint: {checkpoint_name}")
    print(f"Evaluating for {args.num_eval_episodes} episodes\n")

    rewards = []
    lengths = []

    with torch.no_grad():
        for ep in tqdm.tqdm(
            range(args.num_eval_episodes), desc="Evaluating", unit="episode"
        ):
            obs, _ = env.reset(seed=random.randint(0, 10_000_000))
            h = None
            episode_reward = 0.0
            episode_length = 0

            while episode_length < args.max_episode_steps:
                x = torch.FloatTensor(obs).unsqueeze(0).to(device)
                policy_dist, _, h = model(x, h)
                action = int(policy_dist.squeeze(0).argmax().item())
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                episode_length += 1
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
            lengths.append(episode_length)
            tqdm.tqdm.write(
                f"Episode {ep + 1}/{args.num_eval_episodes}: "
                f"Reward = {episode_reward:.2f}, Length = {episode_length}"
            )

    env.close()

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"  Min reward:  {np.min(rewards):.2f}")
    print(f"  Max reward:  {np.max(rewards):.2f}")
    print("=" * 60)
