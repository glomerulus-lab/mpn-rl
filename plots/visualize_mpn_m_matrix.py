"""
Visualize the M matrix (Hebbian plasticity state) of a trained MPN model
across multiple episodes as a GIF. Each frame is a heatmap of the M weights.

Usage:
    python visualize_mpn_m_matrix.py \
        --experiment a2c_run3-gonogo-mpn-h256-l2-44654a07 \
        --num-episodes 5 \
        --output figures/mpn_m_matrix_gonogo.gif
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import neurogym  # noqa: F401
import numpy as np
import torch

from envs import TrialEndWrapper
from nn.mpn import MPN


class _ActorCriticNet(torch.nn.Module):
    """Flexible ActorCriticNet that auto-matches the checkpoint's actor/critic heads."""

    def __init__(
        self,
        input_dim,
        action_dim,
        hidden_dim,
        num_layers,
        activation,
        lambda_max,
        eta_init,
        lambda_init,
        mpn_bias,
        actor_hidden_sizes,
        critic_hidden_sizes,
    ):
        super().__init__()
        self.num_layers = num_layers

        self.core = torch.nn.ModuleList(
            [
                MPN(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    activation=activation,
                    lambda_max=lambda_max,
                    eta_init=eta_init,
                    lambda_init=lambda_init,
                    freeze_plasticity=False,
                    bias=mpn_bias,
                )
                for i in range(num_layers)
            ]
        )

        self.postprocessor = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64), torch.nn.ReLU()
        )

        actor_layers = []
        in_dim = 64
        for out_dim in actor_hidden_sizes:
            actor_layers.append(torch.nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.actor = torch.nn.Sequential(*actor_layers)

        critic_layers = []
        in_dim = 64
        for out_dim in critic_hidden_sizes:
            critic_layers.append(torch.nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.critic = torch.nn.Sequential(*critic_layers)

    def forward(self, x, h):
        if h is None:
            h = [None] * self.num_layers
        new_h = []
        out = x
        for layer, h_i in zip(self.core, h):
            out, h_i_new = layer(out, h_i)
            new_h.append(h_i_new)
        out = self.postprocessor(out)
        logits = self.actor(out)
        policy = torch.nn.functional.softmax(logits, dim=-1)
        value = self.critic(out)
        return policy, value, new_h


def _infer_head_sizes(state_dict, prefix):
    """Read Linear layer sizes for actor or critic from checkpoint keys."""
    idx = 0
    sizes = []
    while f"{prefix}.{idx}.weight" in state_dict:
        w = state_dict[f"{prefix}.{idx}.weight"]
        sizes.append(w.shape[0])  # output dim
        idx += 1
    return sizes  # e.g. [64, 2] for actor


def _make_env(config):
    env = neurogym.make(config["env_name"], **config.get("env_kwargs", {}))
    for key in ("fail", "miss"):
        if key in env.unwrapped.rewards:
            env.unwrapped.rewards[key] = -1.0
    return TrialEndWrapper(env)


def _load_model(exp_dir: Path, device: torch.device):
    config_path = exp_dir / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    env = _make_env(config)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    ckpt = torch.load(
        exp_dir / "checkpoints" / "best_model.pt",
        map_location=device,
        weights_only=False,
    )
    sd = ckpt["model_state_dict"]

    actor_sizes = _infer_head_sizes(sd, "actor")
    critic_sizes = _infer_head_sizes(sd, "critic")

    model = _ActorCriticNet(
        input_dim=input_dim,
        action_dim=action_dim,
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 1),
        activation=config.get("activation", "tanh"),
        lambda_max=config.get("lambda_max", 0.99),
        eta_init=config.get("eta_init", 0.01),
        lambda_init=config.get("lambda_init", 0.99),
        mpn_bias=config.get("mpn_bias", True),
        actor_hidden_sizes=actor_sizes,
        critic_hidden_sizes=critic_sizes,
    ).to(device)

    model.load_state_dict(sd, strict=True)
    model.eval()

    return model, config


def _run_episode(model, env, device, seed=None):
    """Run one episode; return list of step dicts and the trial type."""
    if seed is not None:
        env.unwrapped.rng = np.random.RandomState(seed)

    obs, _ = env.reset()
    h = None
    steps = []
    trial_type = "No-Go"  # default; updated when a Go response is rewarded

    with torch.no_grad():
        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            policy_dist, value, h_new = model(obs_t, h)
            action = int(policy_dist.argmax(-1).item())
            obs_next, reward, terminated, truncated, info = env.step(action)

            # Detect Go trial: agent responded (action=1) and got rewarded
            if action == 1 and reward > 0:
                trial_type = "Go"

            m_matrices = [m_i.squeeze(0).cpu().numpy() for m_i in h_new]
            steps.append(
                {
                    "obs": obs.copy(),
                    "M": m_matrices,
                    "action": action,
                    "reward": reward,
                    "info": info,
                }
            )

            obs = obs_next
            h = h_new
            done = terminated or truncated

    total_reward = sum(s["reward"] for s in steps)
    return steps, total_reward, trial_type


def _render_frame(
    all_episode_lengths,
    all_trial_types,
    layer_norms,
    layer_cmap,
    current_ep,
    num_episodes,
    step_idx,
    step_data,
    num_layers,
    fig,
    axes,
    im_list,
    ax_action,
    action_line,
    cursor_action,
    ax_obs,
    obs_lines,
    cursor_obs,
    all_steps,
):
    """Update figure in-place; return RGB array."""
    M_list = step_data["M"]
    action = step_data["action"]
    reward = step_data["reward"]
    trial_type = all_trial_types[current_ep]

    ep_start = sum(all_episode_lengths[:current_ep])
    ep_step = step_idx - ep_start
    ep_len = all_episode_lengths[current_ep]

    # --- M matrix images (fixed colorscale baked into RGBA pixels) ---
    for layer_idx, (ax, im) in enumerate(zip(axes, im_list)):
        M = M_list[layer_idx]
        rgba = layer_cmap(layer_norms[layer_idx](M))
        im.set_data(rgba)
        ax.set_title(
            f"Layer {layer_idx}  ({M.shape[0]}×{M.shape[1]})", fontsize=9, pad=3
        )

    # --- Action timeline (history up to current step) ---
    ep_steps = all_steps[ep_start : ep_start + ep_step + 1]
    xs = list(range(ep_step + 1))
    ys = [s["action"] for s in ep_steps]
    action_line.set_data(xs, ys)
    cursor_action.set_xdata([ep_step])
    ax_action.set_xlim(-0.5, ep_len - 0.5)

    # --- Observation timeline ---
    for i, line in enumerate(obs_lines):
        line.set_data(xs, [s["obs"][i] for s in ep_steps])
    cursor_obs.set_xdata([ep_step])
    ax_obs.set_xlim(-0.5, ep_len - 0.5)
    all_obs = np.array([s["obs"] for s in all_steps[ep_start : ep_start + ep_len]])
    obs_min, obs_max = all_obs.min() - 0.1, all_obs.max() + 0.1
    ax_obs.set_ylim(obs_min, obs_max)

    action_label = "Respond" if action == 1 else "Fixate"
    fig.suptitle(
        f"Trial {current_ep + 1}/{num_episodes}  [{trial_type} trial]  "
        f"step {ep_step + 1}/{ep_len}  |  "
        f"action={action_label}  reward={reward:+.1f}",
        fontsize=9,
        y=0.99,
    )

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    return buf[:, :, :3]  # drop alpha


def make_gif(
    experiment_dir: str,
    num_episodes: int = 5,
    output_path: str = "mpn_m_matrix.gif",
    fps: int = 3,
    device_str: str = "cpu",
):

    device = torch.device(device_str)
    exp_dir = Path("experiments") / experiment_dir

    print(f"Loading model from {exp_dir} ...")
    model, config = _load_model(exp_dir, device)
    num_layers = config.get("num_layers", 1)
    hidden_dim = config.get("hidden_dim", 128)

    print(f"  env={config['env_name']}  layers={num_layers}  hidden={hidden_dim}")

    # --- Collect all episodes ---
    all_steps = []
    all_episode_lengths = []
    all_trial_types = []
    all_rewards = []

    # Sample episodes ensuring a mix of Go and No-Go trials
    print(f"Running {num_episodes} episodes (seeking Go/No-Go mix) ...")
    go_count = 0
    nogo_count = 0
    seed = 0
    while len(all_episode_lengths) < num_episodes:
        env = _make_env(config)
        steps, total_reward, trial_type = _run_episode(model, env, device, seed=seed)
        env.close()
        seed += 1

        is_go = trial_type == "Go"
        need_go = go_count < (num_episodes + 1) // 2
        need_nogo = nogo_count < num_episodes // 2
        if (is_go and need_go) or (not is_go and need_nogo) or seed > 50:
            ep_num = len(all_episode_lengths) + 1
            print(
                f"  Episode {ep_num}: {len(steps)} steps  reward={total_reward:.2f}  [{trial_type}]"
            )
            all_steps.extend(steps)
            all_episode_lengths.append(len(steps))
            all_trial_types.append(trial_type)
            all_rewards.append(total_reward)
            if is_go:
                go_count += 1
            else:
                nogo_count += 1

    print(f"\nTotal frames: {len(all_steps)}")

    # --- Build figure layout ---
    layer_shapes = [m.shape for m in all_steps[0]["M"]]
    widths = [max(1, s[1] / 20) for s in layer_shapes]
    total_width = sum(widths) + 0.5 * len(widths)
    fig_w = max(6, min(16, total_width + 2))
    fig_h = 6.0  # extra height for action panel

    gs = gridspec.GridSpec(
        3,
        num_layers,
        height_ratios=[4, 1, 1],
        width_ratios=widths,
        hspace=0.55,
        wspace=0.3,
    )
    fig = plt.figure(figsize=(fig_w, fig_h + 1.5))

    axes = [fig.add_subplot(gs[0, i]) for i in range(num_layers)]
    ax_action = fig.add_subplot(gs[1, :])
    ax_obs = fig.add_subplot(gs[2, :])

    # Fixed colorscale — computed once across all frames
    global_vmax = [
        max(max(np.abs(s["M"][li]).max() for s in all_steps), 1e-8)
        for li in range(num_layers)
    ]
    layer_cmap = matplotlib.colormaps["RdBu_r"]
    layer_norms = [mcolors.Normalize(vmin=-vmax, vmax=vmax) for vmax in global_vmax]

    # Initialise heatmap imshow objects with pre-baked RGBA (no vmin/vmax on imshow)
    im_list = []
    for layer_idx, ax in enumerate(axes):
        M0 = all_steps[0]["M"][layer_idx]
        rgba0 = layer_cmap(layer_norms[layer_idx](M0))
        im = ax.imshow(rgba0, aspect="auto", interpolation="nearest")
        ax.axis("off")
        ax.set_title(
            f"Layer {layer_idx}  ({M0.shape[0]}×{M0.shape[1]})", fontsize=9, pad=3
        )
        im_list.append(im)

    max_ep_len = max(all_episode_lengths)
    obs_dim = all_steps[0]["obs"].shape[0]
    obs_labels = [f"obs[{i}]" for i in range(obs_dim)]
    obs_colors = ["#e06c75", "#61afef", "#98c379"]  # red, blue, green

    # Action timeline
    (action_line,) = ax_action.step([], [], where="post", color="steelblue", lw=2)
    cursor_action = ax_action.axvline(x=0, color="tomato", lw=1.5, linestyle="--")
    ax_action.set_xlim(-0.5, max_ep_len - 0.5)
    ax_action.set_ylim(-0.2, 1.3)
    ax_action.set_yticks([0, 1])
    ax_action.set_yticklabels(["No-Go", "Go"], fontsize=8)
    ax_action.set_xlabel("Step within trial", fontsize=8)
    ax_action.set_title("Agent action", fontsize=8, pad=2)
    ax_action.tick_params(labelsize=7)
    ax_action.grid(axis="y", linestyle=":", alpha=0.5)

    # Observation timeline
    obs_lines = [
        ax_obs.step(
            [],
            [],
            where="post",
            color=obs_colors[i % len(obs_colors)],
            lw=1.5,
            label=obs_labels[i],
        )[0]
        for i in range(obs_dim)
    ]
    cursor_obs = ax_obs.axvline(x=0, color="tomato", lw=1.5, linestyle="--")
    ax_obs.set_xlim(-0.5, max_ep_len - 0.5)
    ax_obs.set_xlabel("Step within trial", fontsize=8)
    ax_obs.set_title("Observations", fontsize=8, pad=2)
    ax_obs.tick_params(labelsize=7)
    ax_obs.legend(fontsize=7, loc="upper right", framealpha=0.7)

    # --- Render frames ---
    frames = []
    current_ep = 0
    ep_boundary_steps = set()
    ep_start = 0
    for ep_len in all_episode_lengths:
        ep_boundary_steps.add(ep_start)
        ep_start += ep_len
    ep_boundary_steps.discard(0)

    print("Rendering frames ...")
    for step_idx, step_data in enumerate(all_steps):
        if step_idx in ep_boundary_steps and step_idx > 0:
            current_ep += 1

        frame = _render_frame(
            all_episode_lengths,
            all_trial_types,
            layer_norms,
            layer_cmap,
            current_ep,
            num_episodes,
            step_idx,
            step_data,
            num_layers,
            fig,
            axes,
            im_list,
            ax_action,
            action_line,
            cursor_action,
            ax_obs,
            obs_lines,
            cursor_obs,
            all_steps,
        )
        frames.append(frame)

    plt.close(fig)

    # --- Write GIF ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing {len(frames)} frames → {output_path} ...")
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"Done!  Rewards: {[f'{r:.2f}' for r in all_rewards]}")
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description="Visualize MPN M matrix as GIF")
    parser.add_argument(
        "--experiment",
        type=str,
        default="a2c_run3-gonogo-mpn-h256-l2-44654a07",
        help="Experiment directory name under experiments/",
    )
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--output", type=str, default="figures/mpn_m_matrix_gonogo.gif")
    parser.add_argument(
        "--fps", type=int, default=3, help="Frames per second for the GIF"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    make_gif(
        experiment_dir=args.experiment,
        num_episodes=args.num_episodes,
        output_path=args.output,
        fps=args.fps,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
