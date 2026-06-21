"""
Main CLI for MPN A2C training.

Commands:
    train-neurogym - Train MPN/LSTM/RNN agent on NeuroGym environments using A2C
    eval           - Evaluate a trained agent
    render         - Render episode(s) to a static plot

Examples:
    python main_a2c.py train-neurogym --env-name GoNogo-v0
    python main_a2c.py train-neurogym --env-name GoNogo-v0 --env-config configs/gonogo.json
    python main_a2c.py eval --experiment-name my-agent --num-eval-episodes 10
    python main_a2c.py render --experiment-name my-agent --output render.png
"""

import json
import math
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal, Union

import matplotlib.pyplot as plt
import neurogym  # noqa: F401 — registers NeuroGym environments
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import wandb

import temporal_order_env  # noqa: F401 — registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0
from envs import (
    TrialEndWrapper,
    _create_env_from_config,
    _load_model_from_config,
)

# Local imports
from evaluation import _evaluate_actorcritic
from model_utils import ExperimentManager, get_device
from models.actor_critic import ActorCriticNet
from oracle_agents import get_oracle_reward


def _compute_returns_episode(rewards, dones, next_value, gamma):
    """List-based return computation matching the reference repo."""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns


@dataclass
class TrainConfig:
    """Train on a NeuroGym environment with episode-based A2C and full BPTT."""

    experiment_name: str | None = None
    experiments_dir: Annotated[
        Path,
        tyro.conf.arg(
            help="Root dir for all experiment output "
            "(config, metrics, checkpoints, plots)"
        ),
    ] = Path("experiments")
    env_name: str = "GoNogo-v0"
    env_config: Annotated[
        str | None,
        tyro.conf.arg(help="Path to JSON file of kwargs passed to neurogym.make()"),
    ] = None
    max_episode_steps: int = 500
    tbptt_len: Annotated[
        int, tyro.conf.arg(help="Truncated BPTT chunk length (0 = full episode)")
    ] = 50
    total_frames: int = 500000
    num_episodes: Annotated[
        int, tyro.conf.arg(help="Stop after N episodes (0 = use total_frames)")
    ] = 0
    model_type: Literal["mpn", "mpn-frozen", "rnn", "lstm"] = "lstm"
    hidden_dim: int = 128
    num_layers: int = 1
    activation: Literal["relu", "tanh", "sigmoid"] = "tanh"
    lambda_max: float = 0.99
    eta_init: float = 0.01
    lambda_init: float = 0.99
    gamma: float = 0.98
    entropy_coef: float = 0.01
    value_coef: float = 1.0
    normalize_advantages: bool = False
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 10.0
    print_freq: Annotated[
        int, tyro.conf.arg(help="Evaluate and log every N episodes")
    ] = 50
    num_eval_episodes: int = 10
    device: str = "cpu"
    tag: str | None = None
    experiment_id: str | None = None
    mpn_bias: Annotated[
        bool,
        tyro.conf.arg(
            help="Use bias term in MPN hidden layer; --no-mpn-bias disables it "
            "(correction W*(M+1) is kept)"
        ),
    ] = True
    wandb: bool = False
    wandb_project: str = "mpn-rl"
    wandb_entity: str | None = None


def train_neurogym(args: TrainConfig):
    """Train on NeuroGym env using episode-based A2C with full BPTT.

    Uses ActorCriticNet (matching the example repo architecture) for proper
    BPTT through each episode.
    Supports rnn, lstm, mpn, mpn-frozen.
    """
    print("=" * 60)
    print("Training with A2C + BPTT on NeuroGym")
    print("=" * 60)

    if args.experiment_id is None:
        args.experiment_id = str(uuid.uuid4())[:8]
    if args.experiment_name is not None:
        args.experiment_name = f"{args.experiment_name}-{args.experiment_id}"

    exp_manager = ExperimentManager(args.experiments_dir, args.experiment_name)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Directory:  {exp_manager.exp_dir}\n")

    env_kwargs = {}
    if args.env_config:
        with open(args.env_config) as f:
            env_kwargs = json.load(f)

    config = asdict(args)
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
    print(f"Model: {args.model_type.upper()}, hidden_dim={args.hidden_dim}")
    print(f"lr={args.learning_rate}, gamma={args.gamma}, value_coef={args.value_coef}")
    print(f"total_frames={args.total_frames}\n")

    model = ActorCriticNet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        core_type=args.model_type,
        activation=args.activation,
        lambda_max=args.lambda_max,
        eta_init=args.eta_init,
        lambda_init=args.lambda_init,
        num_layers=args.num_layers,
        mpn_bias=args.mpn_bias,
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
        h = None
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
            policy_dist, value, h = model(obs_t, h)
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
                    _, next_value, _ = model(obs_t, h)
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

                # Detach h to cut gradient graph between chunks
                if isinstance(h, tuple):
                    h = tuple(x.detach() for x in h)
                elif isinstance(h, list):
                    h = [x.detach() for x in h]
                elif h is not None:
                    h = h.detach()

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

        if episode - last_eval_episode >= args.print_freq:
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
                0.0,
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
        h = None
        for _ in tqdm.tqdm(
            range(args.max_episode_steps), desc="Recording", unit="step"
        ):
            observations.append(obs.copy())
            x = torch.FloatTensor(obs).unsqueeze(0)
            policy_dist, _, h = model(x, h)
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


Command = Union[
    Annotated[TrainConfig, tyro.conf.subcommand("train-neurogym")],
    Annotated[EvalConfig, tyro.conf.subcommand("eval")],
    Annotated[RenderConfig, tyro.conf.subcommand("render")],
]


def main():
    cfg = tyro.cli(Command, description="MPN A2C training, evaluation, and rendering.")
    if isinstance(cfg, TrainConfig):
        train_neurogym(cfg)
    elif isinstance(cfg, EvalConfig):
        evaluate(cfg)
    else:
        render_to_plot(cfg)


if __name__ == "__main__":
    main()
