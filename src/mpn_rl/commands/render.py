from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
import tyro

from mpn_rl.envs import _create_env_from_config, _load_model_from_config
from mpn_rl.experiment import ExperimentManager


@dataclass
class RenderConfig:
    """Render an episode to a static plot."""

    experiment_name: str
    experiments_dir: Annotated[
        Path, tyro.conf.arg(help="Root dir the experiment was logged under")
    ] = Path("experiments")
    output: str | None = None
    checkpoint: str | None = None
    max_episode_steps: int = 500


def render_to_plot(args: RenderConfig):
    """Render episode to static plot."""
    print("=" * 60)
    print("Rendering Agent Episode")
    print("=" * 60)

    exp_manager = ExperimentManager(args.experiments_dir, args.experiment_name)
    config = exp_manager.load_config()

    device = torch.device("cpu")
    print(
        f"Model type: {config.get('model_type', 'lstm').upper()}, using CPU for rendering\n"
    )

    model = _load_model_from_config(config, device)
    checkpoint_name = args.checkpoint if args.checkpoint else "best_model.pt"
    exp_manager.load_model(model, checkpoint_name=checkpoint_name, device="cpu")
    model.eval()

    env = _create_env_from_config(config)
    action_dim = env.action_space.n
    output_path = (
        Path(args.output)
        if args.output
        else exp_manager.plot_dir / "episode_render.png"
    )

    observations, actions, rewards = [], [], []

    with torch.no_grad():
        obs, _ = env.reset(seed=0)
        state = None
        for _ in tqdm.tqdm(
            range(args.max_episode_steps), desc="Recording", unit="step"
        ):
            observations.append(obs.copy())
            x = torch.FloatTensor(obs).unsqueeze(0)
            policy_dist, _, state = model(x, state)
            action = int(policy_dist.squeeze(0).argmax().item())
            obs, reward, terminated, truncated, _ = env.step(action)
            actions.append(action)
            rewards.append(reward)
            if terminated or truncated:
                break

    env.close()

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)

    fig, axs = plt.subplots(3, 1, figsize=(12, 8))
    axs[0].plot(observations)
    axs[0].set_title("Observations")
    axs[0].legend(
        [f"Obs {i}" for i in range(observations.shape[1])],
        loc="upper right",
        ncol=observations.shape[1],
    )

    axs[1].step(range(len(actions)), actions, where="post")
    axs[1].set_title("Actions")
    axs[1].set_ylabel("Action")
    axs[1].set_ylim(-0.5, action_dim - 0.5)

    axs[2].plot(rewards)
    axs[2].set_title(f"Rewards (Total: {np.sum(rewards):.2f})")
    axs[2].set_xlabel("Timestep")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")
    print(f"Episode reward: {np.sum(rewards):.2f}, length: {len(rewards)}")
