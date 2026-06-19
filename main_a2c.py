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

import argparse
import json
import math
import random
import uuid
from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import neurogym  # noqa: F401 — registers NeuroGym environments
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb

import temporal_order_env  # registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0

# Local imports
from model_utils import ExperimentManager
from mpn_module import MPN
from oracle_agents import get_oracle_reward


class TrialEndWrapper(gymnasium.Wrapper):
    """End the episode when a new trial starts; normalize correct-withhold reward to +1."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("new_trial", False):
            terminated = True
            # Correct No-go (withheld correctly) gives 0 reward by default; normalize to +1
            if info.get("performance") == 1 and reward == 0.0:
                reward = 1.0
        return obs, reward, terminated, truncated, info


def get_device(device_str="cpu"):
    """Get PyTorch device."""
    if device_str == "gpu":
        device_str = "cuda"

    device = torch.device(device_str)

    if device.type == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(device.index or 0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(device.index or 0).total_memory / 1e9:.2f} GB"
        )
    elif device.type == "cuda":
        print("Warning: GPU requested but not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        print("Using CPU")

    return device


def compute_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    bootstrap_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted returns with episode-boundary handling.

    Args:
        rewards:         [T] reward at each step
        dones:           [T] float done flag (1.0 = episode ended after this step)
        bootstrap_value: scalar — V(s_T) * (1 - done_T), used to bootstrap
                         the last partial episode in the batch
        gamma:           discount factor

    Returns:
        returns: [T] discounted return for each step
    """
    returns = []
    R = bootstrap_value
    for t in reversed(range(len(rewards))):
        R = rewards[t] + gamma * R * (1.0 - dones[t])
        returns.insert(0, R)
    return torch.stack(returns)


def _create_env_from_config(config, device="cpu", max_episode_steps=500):
    """Rebuild the environment used during training from a saved config dict."""
    env_name = config["env_name"]
    env = neurogym.make(env_name, **config.get("env_kwargs", {}))
    for key in ("fail", "miss"):
        if key in env.unwrapped.rewards:
            env.unwrapped.rewards[key] = -1.0
    return TrialEndWrapper(env)


def _load_model_from_config(config, device):
    """Reconstruct an ActorCriticNet from a saved experiment config."""
    env = _create_env_from_config(config, device)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()
    model = ActorCriticNet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=config.get("hidden_dim", 128),
        core_type=config.get("model_type", "lstm"),
        activation=config.get("activation", "tanh"),
        lambda_max=config.get("lambda_max", 0.99),
        eta_init=config.get("eta_init", 0.01),
        lambda_init=config.get("lambda_init", 0.99),
        num_layers=config.get("num_layers", 1),
        mpn_bias=config.get("mpn_bias", True),
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Architecture matching the scale-invariant memory A2C repo
# ---------------------------------------------------------------------------


class ActorCriticNet(nn.Module):
    """Actor-critic network matching the reference A2C architecture.

    Structure:
        core (RNN / LSTM / MPN)
          → postprocessor: Linear(hidden_dim, 64) + ReLU
          → actor:  Linear(64, 64) → Linear(64, action_dim) → Softmax
          → critic: Linear(64, 64) → Linear(64, 1)

    forward(x, h) → (policy_dist, value, h_new)
        x:    (batch, input_dim)
        h:    hidden state (None to reset)
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        core_type: str = "lstm",
        activation: str = "tanh",
        lambda_max: float = 0.99,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
        num_layers: int = 1,
        mpn_bias: bool = True,
        lstm_forget_bias: float = 0.0,
    ):
        super().__init__()
        self.core_type = core_type
        self.num_layers = num_layers

        if core_type == "rnn":
            self.core = nn.RNN(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
        elif core_type == "lstm":
            self.core = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
            if lstm_forget_bias != 0.0:
                for name, param in self.core.named_parameters():
                    if "bias" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(lstm_forget_bias)
        elif core_type in ("mpn", "mpn-frozen"):
            freeze = core_type == "mpn-frozen"
            self.core = nn.ModuleList(
                [
                    MPN(
                        input_dim if i == 0 else hidden_dim,
                        hidden_dim,
                        activation=activation,
                        lambda_max=lambda_max,
                        eta_init=eta_init,
                        lambda_init=lambda_init,
                        freeze_plasticity=freeze,
                        bias=mpn_bias,
                    )
                    for i in range(num_layers)
                ]
            )
        else:
            raise ValueError(f"Unknown core_type: {core_type!r}")

        self.postprocessor = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU())
        self.actor = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, h):
        if self.core_type in ("rnn", "lstm"):
            out, h = self.core(x.unsqueeze(1), h)  # (batch, 1, hidden)
            out = out.squeeze(1)  # (batch, hidden)
        else:  # MPN / MPN-frozen — h is a list of M matrices, one per layer
            if h is None:
                h = [None] * self.num_layers
            new_h = []
            out = x
            for layer, h_i in zip(self.core, h):
                out, h_i_new = layer(out, h_i)
                new_h.append(h_i_new)
            h = new_h
        out = self.postprocessor(out)
        return self.actor(out), self.critic(out), h


# ---------------------------------------------------------------------------
# Episode-based training matching the reference A2C loop
# ---------------------------------------------------------------------------


def _compute_returns_episode(rewards, dones, next_value, gamma):
    """List-based return computation matching the reference repo."""
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * (1 - dones[step])
        returns.insert(0, R)
    return returns


# ---------------------------------------------------------------------------
# Episode-based BPTT training on NeuroGym environments
# ---------------------------------------------------------------------------


def _evaluate_actorcritic(model, env_factory, num_episodes, max_steps, seed, device):
    """Evaluate ActorCriticNet greedily on a fresh env from env_factory."""
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
    return float(np.mean(rewards)), float(np.std(rewards))


def train_neurogym(args):
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

    exp_manager = ExperimentManager(args.experiment_name)
    print(f"Experiment: {exp_manager.experiment_name}")
    print(f"Directory:  {exp_manager.exp_dir}\n")

    env_kwargs = {}
    if args.env_config:
        with open(args.env_config) as f:
            env_kwargs = json.load(f)

    config = vars(args)
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

            eval_reward, eval_reward_std = _evaluate_actorcritic(
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


def evaluate(args):
    """Evaluate trained agent."""
    print("=" * 60)
    print("Evaluating Agent")
    print("=" * 60)

    exp_manager = ExperimentManager(args.experiment_name)
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
                f"Episode {ep+1}/{args.num_eval_episodes}: "
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


def render_to_plot(args):
    """Render episode to static plot."""
    print("=" * 60)
    print("Rendering Agent Episode")
    print("=" * 60)

    exp_manager = ExperimentManager(args.experiment_name)
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


def main():
    parser = argparse.ArgumentParser(description="MPN A2C with TorchRL")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # ------------------------------------------------------------------ #
    # train-neurogym (episode-based A2C with full BPTT)                   #
    # ------------------------------------------------------------------ #
    train_parser = subparsers.add_parser(
        "train-neurogym",
        help="Train on NeuroGym environment with episode-based A2C and full BPTT",
    )
    train_parser.add_argument("--experiment-name", type=str, default=None)
    train_parser.add_argument("--env-name", type=str, default="GoNogo-v0")
    train_parser.add_argument(
        "--env-config",
        type=str,
        default=None,
        help="Path to JSON file of kwargs passed to neurogym.make()",
    )
    train_parser.add_argument("--max-episode-steps", type=int, default=500)
    train_parser.add_argument(
        "--tbptt-len",
        type=int,
        default=50,
        help="Truncated BPTT chunk length (0 = full episode)",
    )
    train_parser.add_argument("--total-frames", type=int, default=500000)
    train_parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Stop after N episodes (0 = use --total-frames)",
    )
    train_parser.add_argument(
        "--model-type",
        type=str,
        default="lstm",
        choices=["mpn", "mpn-frozen", "rnn", "lstm"],
    )
    train_parser.add_argument("--hidden-dim", type=int, default=128)
    train_parser.add_argument("--num-layers", type=int, default=1)
    train_parser.add_argument(
        "--activation", type=str, default="tanh", choices=["relu", "tanh", "sigmoid"]
    )
    train_parser.add_argument("--lambda-max", type=float, default=0.99)
    train_parser.add_argument("--eta-init", type=float, default=0.01)
    train_parser.add_argument("--lambda-init", type=float, default=0.99)
    train_parser.add_argument("--gamma", type=float, default=0.98)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=1.0)
    train_parser.add_argument(
        "--normalize-advantages", action="store_true", default=False
    )
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.0)
    train_parser.add_argument("--grad-clip", type=float, default=10.0)
    train_parser.add_argument(
        "--print-freq", type=int, default=50, help="Evaluate and log every N episodes"
    )
    train_parser.add_argument("--num-eval-episodes", type=int, default=10)
    train_parser.add_argument("--device", type=str, default="cpu")
    train_parser.add_argument("--tag", type=str, default=None)
    train_parser.add_argument("--experiment-id", type=str, default=None)
    train_parser.add_argument(
        "--mpn-bias",
        action="store_true",
        default=True,
        help="Use bias term in MPN hidden layer (default: True)",
    )
    train_parser.add_argument(
        "--no-mpn-bias",
        dest="mpn_bias",
        action="store_false",
        help="Disable bias term in MPN hidden layer; correction W*(M+1) is kept",
    )
    train_parser.add_argument("--wandb", action="store_true", default=False)
    train_parser.add_argument("--wandb-project", type=str, default="mpn-rl")
    train_parser.add_argument("--wandb-entity", type=str, default=None)

    # ------------------------------------------------------------------ #
    # eval                                                                 #
    # ------------------------------------------------------------------ #
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained agent")
    eval_parser.add_argument("--experiment-name", type=str, required=True)
    eval_parser.add_argument("--num-eval-episodes", type=int, default=10)
    eval_parser.add_argument("--checkpoint", type=str, default=None)
    eval_parser.add_argument("--max-episode-steps", type=int, default=500)
    eval_parser.add_argument("--device", type=str, default="cpu")

    # ------------------------------------------------------------------ #
    # render                                                               #
    # ------------------------------------------------------------------ #
    render_parser = subparsers.add_parser("render", help="Render episode to plot")
    render_parser.add_argument("--experiment-name", type=str, required=True)
    render_parser.add_argument("--output", type=str, default=None)
    render_parser.add_argument("--checkpoint", type=str, default=None)
    render_parser.add_argument("--max-episode-steps", type=int, default=500)

    args = parser.parse_args()

    if args.command == "train-neurogym":
        train_neurogym(args)
    elif args.command == "eval":
        evaluate(args)
    elif args.command == "render":
        render_to_plot(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
