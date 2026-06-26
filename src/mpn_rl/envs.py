"""Environment construction and model reconstruction from saved experiment configs."""

import gymnasium
import neurogym  # noqa: F401 — registers NeuroGym environments

import mpn_rl.temporal_order_env  # noqa: F401 — registers TemporalOrder-v0 / TemporalOrder10-v0 / TemporalOrder20-v0
from mpn_rl.models.actor_critic import ActorCriticNet


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
        model_type=config.get("model_type", "lstm"),
        activation=config.get("activation", "tanh"),
        lambda_max=config.get("lambda_max", 0.99),
        eta_init=config.get("eta_init", 0.01),
        lambda_init=config.get("lambda_init", 0.99),
        num_layers=config.get("num_layers", 1),
        mpn_bias=config.get("mpn_bias", True),
        random_proj_dim=config.get("random_proj_dim"),
    ).to(device)
    return model
