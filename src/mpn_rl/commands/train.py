import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Union

import neurogym  # noqa: F401 — registers NeuroGym environments
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import wandb
import yaml
from pydantic import BaseModel, Field

import mpn_rl
import mpn_rl.temporal_order_env  # noqa: F401 — registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0
from mpn_rl.device import get_device
from mpn_rl.envs import TrialEndWrapper
from mpn_rl.evaluation import _evaluate_actorcritic
from mpn_rl.experiment import ExperimentManager
from mpn_rl.models.actor_critic import ActorCriticNet
from mpn_rl.oracle_agents import get_oracle_reward


def _seed_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch global RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compute_returns_episode(rewards, dones, next_value, gamma):
    """List-based return computation matching the reference repo."""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns


# protected_namespaces=() silences pydantic's warning about the `model_type`
# field colliding with its `model_` protected namespace, which we don't use.
class ModelConfig(BaseModel, protected_namespaces=(), extra="forbid"):
    pass


class LSTMConfig(ModelConfig):
    model_type: Literal["lstm"] = "lstm"


class RNNConfig(ModelConfig):
    model_type: Literal["rnn"] = "rnn"


class MPNConfig(ModelConfig):
    model_type: Literal["mpn"] = "mpn"
    eta_init: float = 0.01
    lambda_init: float = Field(0.99, ge=0, le=1)
    lambda_max: float = Field(0.99, ge=0, le=1)
    activation: Literal["relu", "tanh", "sigmoid"] = "tanh"
    mpn_bias: bool = True


class MPNFrozenConfig(ModelConfig):
    model_type: Literal["mpn-frozen"] = "mpn-frozen"
    activation: Literal["relu", "tanh", "sigmoid"] = "tanh"
    mpn_bias: bool = True


Model = Annotated[
    Union[
        Annotated[LSTMConfig, tyro.conf.subcommand("lstm")],
        Annotated[RNNConfig, tyro.conf.subcommand("rnn")],
        Annotated[MPNConfig, tyro.conf.subcommand("mpn")],
        Annotated[MPNFrozenConfig, tyro.conf.subcommand("mpn-frozen")],
    ],
    Field(discriminator="model_type"),
]


class TrainConfig(BaseModel, extra="forbid"):
    """Train on a NeuroGym environment with episode-based A2C and full BPTT."""

    # Identity / output
    sweep_name: str | None = None
    experiment_name: str | None = None
    experiments_dir: Annotated[
        Path,
        tyro.conf.arg(
            help="Root dir for all experiment output "
            "(config, metrics, checkpoints, plots)"
        ),
    ] = Path("experiments")

    # Environment
    env_name: str = "GoNogo-v0"
    env_config: Annotated[
        str | None,
        tyro.conf.arg(help="Path to JSON file of kwargs passed to neurogym.make()"),
    ] = None

    # Rollout / budget — units differ: episodes vs frames (A2C-specific)
    max_episode_steps: int = Field(500, ge=1)
    tbptt_len: Annotated[
        int,
        tyro.conf.arg(help="Truncated BPTT chunk length (0 = full episode)"),
        Field(ge=0),
    ] = 50
    total_frames: int = Field(500000, ge=1)
    num_episodes: Annotated[
        int,
        tyro.conf.arg(help="Stop after N episodes (0 = use total_frames)"),
        Field(ge=0),
    ] = 0

    # Architecture
    hidden_dim: int = Field(128, ge=1)
    num_layers: int = Field(1, ge=1)
    random_proj_dim: Annotated[
        int | None,
        tyro.conf.arg(
            help="Project observations through a fixed, untrained random linear "
            "layer of this dimension before the core (the MPN paper uses 10). "
            "None = feed raw observations."
        ),
        Field(ge=1),
    ] = None
    model: Model = Field(default_factory=LSTMConfig)

    # A2C objective
    gamma: float = Field(0.98, ge=0, le=1)
    entropy_coef: float = Field(0.01, ge=0)
    value_coef: float = Field(1.0, ge=0)
    normalize_advantages: bool = False

    # Optimizer
    learning_rate: float = Field(1e-4, gt=0)
    weight_decay: float = Field(0.0, ge=0)  # L2 (Adam)
    grad_clip: float = Field(10.0, gt=0)

    # Evaluation / logging cadence
    eval_every_n_episodes: Annotated[
        int,
        tyro.conf.arg(help="Evaluate/log/checkpoint every N episodes"),
        Field(ge=1),
    ] = 50
    num_eval_episodes: int = Field(10, ge=1)

    # Reproducibility / device
    seed: int = Field(42, ge=0, lt=2**32)
    device: Literal["cpu", "gpu"] = "cpu"

    # Experiment tracking (wandb)
    tag: str | None = None
    wandb: bool = False
    wandb_project: str = "mpn-rl"
    wandb_entity: str | None = None


@dataclass
class TrainCommand:
    config: Annotated[
        Path | None,
        tyro.conf.arg(
            help="YAML file of TrainConfig fields; CLI flags override its values"
        ),
    ] = None
    train_config: tyro.conf.OmitSubcommandPrefixes[
        tyro.conf.OmitArgPrefixes[TrainConfig]
    ] = field(default_factory=TrainConfig)


def load_train_config(config_path: Path | None) -> TrainConfig:
    if config_path is None:
        return TrainConfig()
    with open(config_path) as f:
        return TrainConfig(**yaml.safe_load(f))


def resolve_train_config(args: list[str]) -> TrainConfig:
    first = tyro.cli(TrainCommand, args=args)
    default = TrainCommand(
        config=first.config, train_config=load_train_config(first.config)
    )
    return tyro.cli(TrainCommand, args=args, default=default).train_config


def train_neurogym(args: TrainConfig):
    """Train on NeuroGym env using episode-based A2C with full BPTT.

    Uses ActorCriticNet (matching the example repo architecture) for proper
    BPTT through each episode.
    Supports rnn, lstm, mpn, mpn-frozen.
    """
    _seed_rngs(args.seed)
    # Single-threaded sequential RL with tiny nets: 1 thread is fastest and keeps
    # a job from oversubscribing its CPU slot (see the Condor env in sweep.py).
    torch.set_num_threads(1)
    print("=" * 60)
    print("Training with A2C + BPTT on NeuroGym")
    print("=" * 60)
    print(f"Code:       {Path(mpn_rl.__file__).parent}")

    exp_manager = ExperimentManager(args.experiments_dir, args.experiment_name)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Directory:  {exp_manager.exp_dir}\n")

    env_kwargs = {}
    if args.env_config:
        with open(args.env_config) as f:
            env_kwargs = json.load(f)

    config = args.model_dump()
    # Keep config.json flat: readers still expect top-level model fields.
    config.update(config.pop("model"))
    config["experiment_name"] = exp_manager.experiment_name
    config["experiment_id"] = exp_manager.experiment_id
    config["experiments_dir"] = str(
        args.experiments_dir
    )  # Path -> str for JSON + wandb
    config["command"] = "train-neurogym"
    config["algorithm"] = "a2c"
    config["env_kwargs"] = env_kwargs
    exp_manager.save_config(config)

    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=exp_manager.experiment_name,
            config=config,
            tags=[args.tag] if args.tag else [],
            dir=str(exp_manager.exp_dir),
        )

    device = get_device(args.device)
    print()

    def make_oracle_env():
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = neurogym.make(args.env_name, **env_kwargs)
        for key in ("fail", "miss"):
            if key in env.unwrapped.rewards:
                env.unwrapped.rewards[key] = -1.0
        return TrialEndWrapper(env)

    def make_train_env():
        env = neurogym.make(args.env_name, **env_kwargs)
        for key in ("fail", "miss"):
            if key in env.unwrapped.rewards:
                env.unwrapped.rewards[key] = -1.0
        return TrialEndWrapper(env)

    # Build one env to get dims, then close it
    _tmp = make_train_env()
    input_dim = _tmp.observation_space.shape[0]
    action_dim = _tmp.action_space.n
    _tmp.close()

    print(f"Environment:  {args.env_name}")
    print(f"Obs dim: {input_dim}, Action dim: {action_dim}")
    print(f"Model: {args.model.model_type.upper()}, hidden_dim={args.hidden_dim}")
    print(f"lr={args.learning_rate}, gamma={args.gamma}, value_coef={args.value_coef}")
    print(f"total_frames={args.total_frames}\n")

    model = ActorCriticNet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        random_proj_dim=args.random_proj_dim,
        **args.model.model_dump(),
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    env = make_train_env()
    ep_rewards: list[float] = []
    eval_rewards_history: list[float] = []
    total_frames = 0
    last_eval_episode = 0
    best_rolling_avg = -float("inf")
    _eval_rng = np.random.default_rng(seed=42)

    print("Starting training...")
    print("-" * 60)

    use_ep_limit = args.num_episodes > 0
    pbar_total = args.num_episodes if use_ep_limit else args.total_frames
    pbar = tqdm.tqdm(
        total=pbar_total,
        desc="Episodes" if use_ep_limit else "Frames",
        unit="ep" if use_ep_limit else "fr",
    )

    episode = 0
    while (use_ep_limit and episode < args.num_episodes) or (
        not use_ep_limit and total_frames < args.total_frames
    ):
        obs, _ = env.reset(seed=random.randint(0, 10_000_000))
        state = None
        ep_reward = 0.0
        actor_loss = critic_loss = total_loss = None

        chunk_log_probs, chunk_values, chunk_rewards, chunk_dones, chunk_entropies = (
            [],
            [],
            [],
            [],
            [],
        )

        for step in range(args.max_episode_steps):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            policy_dist, value, state = model(obs_t, state)
            action = int(
                np.random.choice(
                    action_dim, p=policy_dist.detach().cpu().numpy().squeeze()
                )
            )
            log_prob = torch.log(policy_dist.squeeze(0)[action])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_frames += 1
            ep_reward += reward

            chunk_log_probs.append(log_prob)
            chunk_values.append(value)
            chunk_rewards.append(reward)
            chunk_dones.append(float(done))
            chunk_entropies.append(
                -(policy_dist * torch.log(policy_dist + 1e-8)).sum(-1)
            )

            obs = next_obs

            chunk_full = args.tbptt_len > 0 and len(chunk_log_probs) == args.tbptt_len
            if chunk_full or done or step == args.max_episode_steps - 1:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                    _, next_value, _ = model(obs_t, state)
                    next_value = next_value.squeeze().item() * (1.0 - float(done))

                returns = _compute_returns_episode(
                    chunk_rewards, chunk_dones, next_value, args.gamma
                )

                log_probs_t = torch.stack(chunk_log_probs)
                returns_t = torch.FloatTensor(returns).to(device)
                values_t = torch.cat(chunk_values).squeeze(-1)
                entropy = torch.stack(chunk_entropies).mean()

                advantages = returns_t - values_t.detach()
                if args.normalize_advantages and advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / advantages.std()

                actor_loss = -(log_probs_t * advantages).mean()
                critic_loss = F.mse_loss(values_t, returns_t.detach())
                total_loss = (
                    actor_loss
                    + args.value_coef * critic_loss
                    - args.entropy_coef * entropy
                )

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                # Detach state to cut gradient graph between chunks
                if isinstance(state, tuple):
                    state = tuple(s.detach() for s in state)
                elif isinstance(state, list):
                    state = [s.detach() for s in state]
                elif state is not None:
                    state = state.detach()

                (
                    chunk_log_probs,
                    chunk_values,
                    chunk_rewards,
                    chunk_dones,
                    chunk_entropies,
                ) = ([], [], [], [], [])

            if done:
                break

        ep_rewards.append(ep_reward)
        episode += 1
        pbar.update((episode if use_ep_limit else total_frames) - pbar.n)

        if episode - last_eval_episode >= args.eval_every_n_episodes:
            last_eval_episode = episode
            eval_seed = int(_eval_rng.integers(0, 2**31))

            eval_reward, eval_reward_std, _ = _evaluate_actorcritic(
                model,
                make_train_env,
                args.num_eval_episodes,
                args.max_episode_steps,
                eval_seed,
                device,
            )
            oracle_reward = get_oracle_reward(
                args.env_name,
                n_episodes=args.num_eval_episodes,
                max_steps=args.max_episode_steps,
                seed=eval_seed,
                env_factory=make_oracle_env,
            )
            pct_oracle = (
                (eval_reward / oracle_reward * 100.0)
                if oracle_reward > 0
                else float("nan")
            )

            avg_ep_reward = float(np.mean(ep_rewards[-100:])) if ep_rewards else 0.0

            last_actor_loss = actor_loss.item() if actor_loss is not None else 0.0
            last_critic_loss = critic_loss.item() if critic_loss is not None else 0.0

            exp_manager.append_training_history(
                total_frames,
                float(eval_reward),
                step + 1,
                last_actor_loss,
                oracle_reward=float(oracle_reward),
                pct_oracle=float(pct_oracle) if not math.isnan(pct_oracle) else None,
                episode=episode,
            )

            tqdm.tqdm.write(
                f"Frames {total_frames:7d} | ep {episode:6d} | "
                f"Eval: {eval_reward:7.2f} ± {eval_reward_std:5.2f} | "
                f"Oracle: {pct_oracle:5.1f}% ({oracle_reward:.1f}) | "
                f"Loss: {last_actor_loss:.4f}"
            )

            if args.wandb:
                wandb.log(
                    {
                        "eval/reward": eval_reward,
                        "eval/reward_std": eval_reward_std,
                        "eval/oracle_reward": oracle_reward,
                        "eval/pct_oracle": pct_oracle,
                        "train/actor_loss": last_actor_loss,
                        "train/critic_loss": last_critic_loss,
                        "train/ep_reward": avg_ep_reward,
                        "train/frames": total_frames,
                        "train/episode": episode,
                    }
                )

            eval_rewards_history.append(eval_reward)
            rolling_avg = float(np.mean(eval_rewards_history[-30:]))
            if rolling_avg > best_rolling_avg:
                best_rolling_avg = rolling_avg
                exp_manager.save_model(
                    model,
                    optimizer=optimizer,
                    checkpoint_name="best_model.pt",
                    metadata={"episode": episode, "eval_reward": rolling_avg},
                )

    env.close()
    pbar.close()

    exp_manager.save_model(
        model,
        optimizer=optimizer,
        checkpoint_name="final_model.pt",
        metadata={"total_frames": args.total_frames, "final": True},
    )

    print("\n" + "=" * 60)
    print("Training complete")
    print(f"Best rolling avg reward (last 30): {best_rolling_avg:.3f}")
    print(f"Results saved to: {exp_manager.exp_dir}")
    print("=" * 60)

    if args.wandb:
        wandb.finish()
