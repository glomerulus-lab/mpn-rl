from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import tyro

from mpn_rl.device import get_device
from mpn_rl.envs import _create_env_from_config, _load_model_from_config
from mpn_rl.evaluation import evaluate_actorcritic, evaluate_supervised
from mpn_rl.experiment import ExperimentManager
from mpn_rl.supervised_data import MaskedSequenceSampler


@dataclass
class EvalConfig:
    """Evaluate a trained agent."""

    experiment_name: str
    experiments_dir: Annotated[
        Path, tyro.conf.arg(help="Root dir the experiment was logged under")
    ] = Path("experiments")
    num_eval_episodes: Annotated[
        int | None,
        tyro.conf.arg(help="a2c: episodes to roll out (default: the config's value)"),
    ] = None
    num_eval_sequences: Annotated[
        int | None,
        tyro.conf.arg(
            help="supervised: sequences to score (default: the config's value)"
        ),
    ] = None
    checkpoint: str | None = None
    max_episode_steps: int = 500
    seed: int | None = None
    device: Literal["cpu", "gpu"] = "cpu"


def evaluate(args: EvalConfig):
    """Evaluate a trained agent, dispatching on the algorithm it was trained with."""
    print("=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    exp_manager = ExperimentManager(args.experiments_dir, args.experiment_name)
    config = exp_manager.load_config()
    algorithm = config.get("algorithm", "a2c")

    device = get_device(args.device)
    print(f"Algorithm:  {algorithm.upper()}")
    print(f"Model type: {config.get('model_type', 'lstm').upper()}\n")

    model = _load_model_from_config(config, device)
    checkpoint_name = args.checkpoint if args.checkpoint else "best_model.pt"
    exp_manager.load_model(model, checkpoint_name=checkpoint_name, device=str(device))
    print(f"Loaded checkpoint: {checkpoint_name}\n")

    if algorithm == "supervised":
        if args.num_eval_episodes is not None:
            raise ValueError(
                f"--num-eval-episodes applies to a2c, but {args.experiment_name} "
                "is a supervised experiment (use --num-eval-sequences)"
            )
        _eval_supervised(args, config, model, device)
    elif algorithm == "a2c":
        if args.num_eval_sequences is not None:
            raise ValueError(
                f"--num-eval-sequences applies to supervised, but "
                f"{args.experiment_name} is an a2c experiment "
                "(use --num-eval-episodes)"
            )
        _eval_actorcritic(args, config, model, device)
    else:
        raise ValueError(f"Unknown algorithm in config: {algorithm!r}")


def _eval_actorcritic(args, config, model, device):
    """Greedy reward rollout: report reward and episode-length statistics."""
    num_episodes = (
        args.num_eval_episodes
        if args.num_eval_episodes is not None
        else config.get("num_eval_episodes", 10)
    )
    print(f"Evaluating for {num_episodes} episodes\n")
    rewards, lengths = evaluate_actorcritic(
        model,
        lambda: _create_env_from_config(config),
        num_episodes,
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


def _eval_supervised(args, config, model, device):
    """Masked per-timestep accuracy over freshly sampled sequences.

    The sequence count defaults to the saved config's num_eval_sequences (what
    training used), overridable with --num-eval-sequences.
    """
    num_sequences = (
        args.num_eval_sequences
        if args.num_eval_sequences is not None
        else config.get("num_eval_sequences", 1000)
    )
    print(f"Evaluating accuracy over {num_sequences} sequences\n")
    # Default to a held-out test stream, distinct from both the training stream
    # (seed) and the validation stream used to select best_model (seed + 10_000).
    sampler = MaskedSequenceSampler(
        config["env_name"],
        config.get("env_kwargs", {}),
        config.get("batch_size", 32),
        config.get("sequence_len", 100),
        seed=args.seed if args.seed is not None else config.get("seed", 42) + 20_000,
    )
    accuracy = evaluate_supervised(model, sampler, num_sequences, device)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Masked accuracy: {accuracy:.4f}  ({num_sequences} sequences)")
    print("=" * 60)
