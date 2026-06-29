"""
Compare trained models from experiments on evaluation metrics.

Available metrics:
- cumulative_reward: Mean and std of episode rewards
- reward_variance: Variance of rewards across episodes
- parameter_count: Total trainable parameters
- worst_vs_best: Worst and best episode rewards with gap

Usage:
    python compare_models.py --env-name GoNogo-v0 --metrics all --num-episodes 20
    python compare_models.py --experiments exp1 exp2 exp3 --num-episodes 10
    python compare_models.py --experiments exp1 exp2 --seeds 42 43 44 45 46
    python compare_models.py --experiments exp1 exp2 --metrics cumulative_reward parameter_count
    python compare_models.py --experiments exp1 exp2 --output results.json
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import neurogym  # noqa: F401 — registers neurogym environments
import numpy as np
import torch

import mpn_rl.temporal_order_env  # noqa: F401 — registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0
from mpn_rl.envs import _create_env_from_config, _load_model_from_config
from mpn_rl.evaluation import evaluate_actorcritic
from mpn_rl.experiment import ExperimentManager

AVAILABLE_METRICS = [
    "cumulative_reward",
    "reward_variance",
    "parameter_count",
    "worst_vs_best",
]

# Config keys excluded from varying hyperparameter detection
EXCLUDED_HPARAM_KEYS = {
    "experiment_name",
    "experiment_id",
    "command",
    "tag",
    "device",
    "checkpoint_freq",
    "max_checkpoints",
    "eval_every_n_episodes",
    "num_eval_episodes",
    "num_envs",
    "frames_per_batch",
    "grad_clip",
    "epsilon_end",
    # Run metadata / logging config injected by main_a2c (not hyperparameters)
    "algorithm",
    "created_at",
    "schema_version",
    "experiments_dir",
    "env_config",
    "wandb",
    "wandb_entity",
    "wandb_project",
    # Already fixed table columns
    "model_type",
    "num_layers",
    "hidden_dim",
    # Environment is the grouping key
    "env_name",
}


def discover_experiments_by_env(
    experiments_dir: str | Path, env_name: str
) -> List[str]:
    """Scan experiments/ for all experiment dirs matching env_name with best_model.pt."""
    base = Path(experiments_dir)
    if not base.exists():
        raise FileNotFoundError(f"Experiments directory not found: {base}")

    matching = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if config.get("env_name") != env_name:
            continue
        checkpoint_path = exp_dir / "checkpoints" / "best_model.pt"
        if not checkpoint_path.exists():
            warnings.warn(f"Skipping {exp_dir.name}: no best_model.pt")
            continue
        matching.append(exp_dir.name)

    return matching


def detect_varying_hyperparameters(configs: Dict[str, Dict]) -> List[str]:
    """Find config keys with >1 unique value across experiments, excluding metadata."""
    if len(configs) < 2:
        return []

    # Collect all keys across configs
    all_keys: Set[str] = set()
    for config in configs.values():
        all_keys.update(config.keys())

    varying = []
    for key in sorted(all_keys):
        if key in EXCLUDED_HPARAM_KEYS:
            continue
        values = set()
        for config in configs.values():
            val = config.get(key)
            # Make unhashable types hashable
            if isinstance(val, (list, dict)):
                val = json.dumps(val, sort_keys=True)
            values.add(val)
        if len(values) > 1:
            varying.append(key)

    return varying


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_experiment(
    experiments_dir: str | Path, experiment_name: str, device: torch.device
) -> Tuple[torch.nn.Module, Dict, str]:
    """
    Load a trained ActorCriticNet from an experiment.

    Returns:
        Tuple of (model, config, model_type)
    """
    exp_manager = ExperimentManager(experiments_dir, experiment_name)
    config = exp_manager.load_config()
    model_type = config.get("model_type", "mpn")

    checkpoint_path = exp_manager.checkpoint_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"No best_model.pt found for experiment '{experiment_name}'"
        )

    model = _load_model_from_config(config, device)
    exp_manager.load_model(model, checkpoint_name="best_model.pt", device=str(device))
    model.eval()

    return model, config, model_type


def evaluate_model_rewards(
    model: torch.nn.Module,
    config: Dict,
    seeds: List[int],
    max_steps: int,
    device: torch.device,
) -> List[float]:
    """Greedily roll out the model once per seed; return per-seed rewards.

    Each seed is run as a single-episode evaluation so the returned list aligns
    one-to-one with ``seeds`` (needed for worst_vs_best / per-seed reporting).
    """
    rewards = []
    for seed in seeds:
        ep_rewards, _ = evaluate_actorcritic(
            model,
            lambda cfg=config: _create_env_from_config(cfg, device=device),
            num_episodes=1,
            max_steps=max_steps,
            seed=seed,
            device=device,
        )
        rewards.append(ep_rewards[0])
    return rewards


def evaluate_random_rewards(
    config: Dict, seeds: List[int], max_steps: int
) -> List[float]:
    """Roll out random actions once per seed; return per-seed rewards.

    Mirrors ``evaluate_model_rewards`` seeding (``env.unwrapped.rng =
    RandomState(seed)``) so the random baseline faces the same trial sequence as
    the trained models.
    """
    rewards = []
    for seed in seeds:
        env = _create_env_from_config(config)
        env.unwrapped.rng = np.random.RandomState(seed)
        action_rng = np.random.RandomState(seed)
        action_dim = env.action_space.n
        obs, _ = env.reset()
        total = 0.0
        for _ in range(max_steps):
            action = int(action_rng.randint(0, action_dim))
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            if terminated or truncated:
                break
        rewards.append(total)
        env.close()
    return rewards


def validate_compatibility(configs: Dict[str, Dict]) -> Tuple[bool, str]:
    """Validate that all experiments have compatible environments."""
    if len(configs) < 2:
        return True, ""

    exp_names = list(configs.keys())
    ref_env_name = configs[exp_names[0]]["env_name"]

    for exp_name in exp_names[1:]:
        config = configs[exp_name]
        if config["env_name"] != ref_env_name:
            return False, (
                f"Environment mismatch: '{exp_names[0]}' uses '{ref_env_name}' "
                f"but '{exp_name}' uses '{config['env_name']}'. "
                f"Models must be trained on the same environment to compare."
            )

    return True, ""


def compute_metrics(
    episode_rewards: List[float],
    model: Optional[torch.nn.Module],
    seeds: List[int],
    requested_metrics: List[str],
) -> Dict[str, Any]:
    """Compute requested metrics from episode rewards."""
    rewards_array = np.array(episode_rewards)
    metrics = {}

    if "cumulative_reward" in requested_metrics:
        metrics["cumulative_reward"] = {
            "mean": float(np.mean(rewards_array)),
            "std": float(np.std(rewards_array)),
            "median": float(np.median(rewards_array)),
        }

    if "reward_variance" in requested_metrics:
        metrics["reward_variance"] = float(np.var(rewards_array))

    if "parameter_count" in requested_metrics:
        if model is not None:
            metrics["parameter_count"] = count_parameters(model)
        else:
            metrics["parameter_count"] = 0

    if "worst_vs_best" in requested_metrics:
        worst_idx = int(np.argmin(rewards_array))
        best_idx = int(np.argmax(rewards_array))
        metrics["worst_vs_best"] = {
            "worst": {"reward": float(np.min(rewards_array)), "seed": seeds[worst_idx]},
            "best": {"reward": float(np.max(rewards_array)), "seed": seeds[best_idx]},
            "gap": float(np.max(rewards_array) - np.min(rewards_array)),
        }

    return metrics


def compare_models(
    experiments_dir: str | Path,
    experiment_names: List[str],
    metrics: List[str],
    num_episodes: int = 10,
    seeds: Optional[List[int]] = None,
    max_steps: int = 500,
    device: str = "cpu",
    varying_hparams: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compare trained models on specified metrics."""
    eval_seeds = seeds if seeds is not None else list(range(42, 42 + num_episodes))
    device = torch.device(device)

    print(f"Comparing {len(experiment_names)} models")
    print(f"Metrics: {metrics}")
    print(f"Episodes: {len(eval_seeds)} (seeds: {eval_seeds})")
    print("=" * 60)

    # Load all experiments first to validate compatibility
    models = {}
    configs = {}
    model_types = {}

    for exp_name in experiment_names:
        print(f"\nLoading: {exp_name}")
        try:
            model, config, model_type = load_model_from_experiment(
                experiments_dir, exp_name, device
            )
            models[exp_name] = model
            configs[exp_name] = config
            model_types[exp_name] = model_type
            print(f"  Type: {model_type.upper()}, Env: {config['env_name']}")
        except FileNotFoundError as e:
            print(f"  ERROR: {e}")
            return {"error": str(e)}

    is_valid, error_msg = validate_compatibility(configs)
    if not is_valid:
        print(f"\nERROR: {error_msg}")
        return {"error": error_msg}

    print("\nModels compatible. Evaluating...")

    ref_config = configs[experiment_names[0]]
    env_name = ref_config["env_name"]

    # Detect varying hyperparameters if not provided
    if varying_hparams is None:
        varying_hparams = detect_varying_hyperparameters(configs)
    if varying_hparams:
        print(f"Varying hyperparameters: {varying_hparams}")

    results = {
        "metadata": {
            "environment": env_name,
            "num_episodes": len(eval_seeds),
            "seeds": eval_seeds,
            "max_steps": max_steps,
            "metrics_computed": metrics,
            "varying_hyperparameters": varying_hparams,
        },
        "models": {},
    }

    needs_episodes = any(
        m in metrics for m in ["cumulative_reward", "reward_variance", "worst_vs_best"]
    )

    # Evaluate random baseline first
    print(f"\n{'=' * 60}")
    print("Model: random (baseline)")
    print(f"{'=' * 60}")

    random_rewards = []
    if needs_episodes:
        random_rewards = evaluate_random_rewards(ref_config, eval_seeds, max_steps)
        for seed, reward in zip(eval_seeds, random_rewards):
            print(f"  Seed {seed:4d}: Reward = {reward:8.2f}")

    random_metrics = compute_metrics(random_rewards, None, eval_seeds, metrics)
    random_metrics["model_type"] = "random"
    random_metrics["hidden_dim"] = 0
    random_metrics["num_layers"] = 0

    if varying_hparams:
        random_metrics["hyperparameters"] = {k: None for k in varying_hparams}

    if needs_episodes:
        random_metrics["episode_rewards"] = random_rewards

    results["models"]["random"] = random_metrics

    print("\n  Results:")
    if "parameter_count" in random_metrics:
        print(f"    Parameters:      {random_metrics['parameter_count']:,}")
    if "cumulative_reward" in random_metrics:
        cr = random_metrics["cumulative_reward"]
        print(f"    Mean Reward:     {cr['mean']:.2f} ± {cr['std']:.2f}")
    if "reward_variance" in random_metrics:
        print(f"    Variance:        {random_metrics['reward_variance']:.2f}")
    if "worst_vs_best" in random_metrics:
        wb = random_metrics["worst_vs_best"]
        print(
            f"    Best Episode:    {wb['best']['reward']:.2f} (seed {wb['best']['seed']})"
        )
        print(
            f"    Worst Episode:   {wb['worst']['reward']:.2f} (seed {wb['worst']['seed']})"
        )
        print(f"    Gap:             {wb['gap']:.2f}")

    # Evaluate each trained model
    for exp_name in experiment_names:
        print(f"\n{'=' * 60}")
        print(f"Model: {exp_name}")
        print(f"{'=' * 60}")

        model = models[exp_name]
        config = configs[exp_name]

        episode_rewards = []

        if needs_episodes:
            episode_rewards = evaluate_model_rewards(
                model, config, eval_seeds, max_steps, device
            )
            for seed, reward in zip(eval_seeds, episode_rewards):
                print(f"  Seed {seed:4d}: Reward = {reward:8.2f}")

        model_metrics = compute_metrics(episode_rewards, model, eval_seeds, metrics)
        model_metrics["model_type"] = model_types[exp_name]
        model_metrics["hidden_dim"] = config["hidden_dim"]
        model_metrics["num_layers"] = config.get("num_layers", 1)

        if varying_hparams:
            model_metrics["hyperparameters"] = {
                k: config.get(k) for k in varying_hparams
            }

        if needs_episodes:
            model_metrics["episode_rewards"] = episode_rewards

        results["models"][exp_name] = model_metrics

        print("\n  Results:")
        if "parameter_count" in model_metrics:
            print(f"    Parameters:      {model_metrics['parameter_count']:,}")
        if "cumulative_reward" in model_metrics:
            cr = model_metrics["cumulative_reward"]
            print(f"    Mean Reward:     {cr['mean']:.2f} ± {cr['std']:.2f}")
        if "reward_variance" in model_metrics:
            print(f"    Variance:        {model_metrics['reward_variance']:.2f}")
        if "worst_vs_best" in model_metrics:
            wb = model_metrics["worst_vs_best"]
            print(
                f"    Best Episode:    {wb['best']['reward']:.2f} (seed {wb['best']['seed']})"
            )
            print(
                f"    Worst Episode:   {wb['worst']['reward']:.2f} (seed {wb['worst']['seed']})"
            )
            print(f"    Gap:             {wb['gap']:.2f}")

    return results


def print_comparison_table(results: Dict[str, Any]):
    """Print a formatted comparison table."""
    if "error" in results:
        print(f"\nError: {results['error']}")
        return

    models = results["models"]
    exp_names = list(models.keys())
    computed_metrics = results["metadata"]["metrics_computed"]
    num_episodes = results["metadata"]["num_episodes"]
    varying_hparams = results["metadata"].get("varying_hyperparameters", [])

    # Get random baseline mean for computing improvement
    random_mean = None
    if "random" in models and "cumulative_reward" in computed_metrics:
        random_mean = models["random"]["cumulative_reward"]["mean"]

    # Build header parts with their widths
    header_parts = [
        f"{'Model':<32}",
        f"{'Type':<12}",
        f"{'Layers':<8}",
        f"{'Hidden':<8}",
    ]
    # Dynamic columns for varying hyperparameters
    for hp in varying_hparams:
        col_width = max(len(hp), 10)
        header_parts.append(f"{hp:<{col_width}}")
    if "parameter_count" in computed_metrics:
        header_parts.append(f"{'Params':<12}")
    if "cumulative_reward" in computed_metrics:
        header_parts.append(f"{'Mean':<10}")
        header_parts.append(f"{'Std':<10}")
    if "reward_variance" in computed_metrics:
        header_parts.append(f"{'Variance':<12}")
    if "cumulative_reward" in computed_metrics and random_mean is not None:
        header_parts.append(f"{'vs Random':<12}")
    if "worst_vs_best" in computed_metrics:
        header_parts.append(f"{'Best':<10}")
        header_parts.append(f"{'Worst':<10}")
        header_parts.append(f"{'Gap':<10}")

    # Calculate table width based on header
    header_line = " ".join(header_parts)
    table_width = len(header_line)

    print("\n" + "=" * table_width)
    print(f"COMPARISON TABLE ({num_episodes} episodes)")
    print("=" * table_width)
    print(header_line)
    print("-" * table_width)

    for exp_name in exp_names:
        m = models[exp_name]
        layers_str = str(m.get("num_layers", 0)) if m.get("num_layers", 0) > 0 else "-"
        hidden_str = str(m.get("hidden_dim", 0)) if m.get("hidden_dim", 0) > 0 else "-"
        row_parts = [
            f"{exp_name:<32}",
            f"{m['model_type']:<12}",
            f"{layers_str:<8}",
            f"{hidden_str:<8}",
        ]

        # Dynamic columns for varying hyperparameters
        hparams = m.get("hyperparameters", {})
        for hp in varying_hparams:
            col_width = max(len(hp), 10)
            val = hparams.get(hp)
            val_str = "-" if val is None else str(val)
            row_parts.append(f"{val_str:<{col_width}}")

        if "parameter_count" in computed_metrics:
            row_parts.append(f"{m['parameter_count']:<12,}")
        if "cumulative_reward" in computed_metrics:
            cr = m["cumulative_reward"]
            row_parts.append(f"{cr['mean']:<10.2f}")
            row_parts.append(f"{cr['std']:<10.2f}")
        if "reward_variance" in computed_metrics:
            row_parts.append(f"{m['reward_variance']:<12.2f}")
        if "cumulative_reward" in computed_metrics and random_mean is not None:
            if exp_name == "random":
                row_parts.append(f"{'-':<12}")
            else:
                cr = m["cumulative_reward"]
                improvement = cr["mean"] - random_mean
                pct = (improvement / abs(random_mean)) * 100 if random_mean != 0 else 0
                row_parts.append(f"{pct:+.1f}%".ljust(12))
        if "worst_vs_best" in computed_metrics:
            wb = m["worst_vs_best"]
            row_parts.append(f"{wb['best']['reward']:<10.2f}")
            row_parts.append(f"{wb['worst']['reward']:<10.2f}")
            row_parts.append(f"{wb['gap']:<10.2f}")

        print(" ".join(row_parts))

    print("=" * table_width)

    # Summary statistics
    if "cumulative_reward" in computed_metrics:
        trained_models = [n for n in exp_names if n != "random"]
        if trained_models:
            print("\nSUMMARY:")
            print("-" * 60)

            best_model = max(
                trained_models, key=lambda x: models[x]["cumulative_reward"]["mean"]
            )
            print(
                f"  Best by mean reward:      {best_model} ({models[best_model]['cumulative_reward']['mean']:.2f})"
            )

            if "reward_variance" in computed_metrics:
                lowest_var = min(
                    trained_models, key=lambda x: models[x]["reward_variance"]
                )
                print(
                    f"  Most consistent:          {lowest_var} (var={models[lowest_var]['reward_variance']:.2f})"
                )

            if "parameter_count" in computed_metrics:
                fewest_params = min(
                    trained_models, key=lambda x: models[x]["parameter_count"]
                )
                print(
                    f"  Fewest parameters:        {fewest_params} ({models[fewest_params]['parameter_count']:,})"
                )

            if "random" in exp_names:
                random_mean = models["random"]["cumulative_reward"]["mean"]
                print(f"\n  Random baseline mean:     {random_mean:.2f}")
                for name in trained_models:
                    improvement = (
                        models[name]["cumulative_reward"]["mean"] - random_mean
                    )
                    pct = (
                        (improvement / abs(random_mean)) * 100
                        if random_mean != 0
                        else 0
                    )
                    print(f"  {name:<24} +{improvement:.2f} ({pct:+.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare trained models on evaluation metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available metrics: {", ".join(AVAILABLE_METRICS)}

Examples:
    # Compare all experiments for an environment
    python compare_models.py --env-name GoNogo-v0 --metrics all

    # Compare with all metrics
    python compare_models.py --experiments exp1 exp2 exp3 --metrics all

    # Compare specific metrics
    python compare_models.py --experiments exp1 exp2 --metrics cumulative_reward parameter_count

    # Use fixed seeds
    python compare_models.py --experiments exp1 exp2 --seeds 42 43 44 45 46

    # Save results to JSON
    python compare_models.py --experiments exp1 exp2 --output results.json
        """,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--experiments", type=str, nargs="+", help="Experiment names to compare"
    )
    source_group.add_argument(
        "--env-name",
        type=str,
        help="Environment name to discover and compare all experiments for",
    )
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=Path("experiments"),
        help="Root dir the experiments were logged under (default: experiments/)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["all"],
        help=f'Metrics to compute: {", ".join(AVAILABLE_METRICS)}, or "all" (default: all)',
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes (default: 10, ignored if --seeds provided)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Fixed seeds for reproducibility",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Max steps per episode (default: 500)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output JSON file path"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", help="Device (default: cpu)"
    )

    args = parser.parse_args()

    # Resolve experiment list
    if args.env_name:
        experiment_names = discover_experiments_by_env(
            args.experiments_dir, args.env_name
        )
        if not experiment_names:
            print(f"No completed experiments found for environment: {args.env_name}")
            return
        print(f"Discovered {len(experiment_names)} experiments for {args.env_name}")
    else:
        experiment_names = args.experiments

    # Auto-generate output path for --env-name when --output not specified
    output_path = args.output
    if output_path is None and args.env_name:
        env_short = args.env_name.replace("-v0", "").lower()
        output_path = f"condor/outputs/compare_{env_short}.json"

    if "all" in args.metrics:
        metrics = AVAILABLE_METRICS
    else:
        metrics = []
        for m in args.metrics:
            if m not in AVAILABLE_METRICS:
                print(f"Unknown metric: {m}")
                print(f"Available: {', '.join(AVAILABLE_METRICS)}")
                return
            metrics.append(m)

    results = compare_models(
        experiments_dir=args.experiments_dir,
        experiment_names=experiment_names,
        metrics=metrics,
        num_episodes=args.num_episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        device=args.device,
    )

    print_comparison_table(results)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
