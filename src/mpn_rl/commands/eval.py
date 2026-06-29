from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import tyro

from mpn_rl.device import get_device
from mpn_rl.envs import _create_env_from_config, _load_model_from_config
from mpn_rl.evaluation import evaluate_actorcritic
from mpn_rl.experiment import ExperimentManager


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
    seed: int | None = None
    device: Literal["cpu", "gpu"] = "cpu"


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

    print(f"Loaded checkpoint: {checkpoint_name}")
    print(f"Evaluating for {args.num_eval_episodes} episodes\n")

    rewards, lengths = evaluate_actorcritic(
        model,
        lambda: _create_env_from_config(config),
        args.num_eval_episodes,
        args.max_episode_steps,
        args.seed,
        device,
        progress=True,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Mean length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(f"  Min reward:  {np.min(rewards):.2f}")
    print(f"  Max reward:  {np.max(rewards):.2f}")
    print("=" * 60)
