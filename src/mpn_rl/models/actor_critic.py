"""Actor-critic network for episode-based A2C training."""

import torch
import torch.nn as nn

from mpn_rl.nn.recurrent_core import RecurrentCore


class ActorCriticNet(nn.Module):
    """Actor-critic network matching the reference A2C architecture.

    Structure:
        RecurrentCore (optional input projection → RNN / LSTM / MPN)
          → postprocessor: Linear(hidden_dim, 64) + ReLU
          → actor:  Linear(64, 64) → Linear(64, action_dim) → Softmax
          → critic: Linear(64, 64) → Linear(64, 1)

    forward(x, state) → (policy_dist, value, new_state)
        x:     (batch, input_dim)
        state: core recurrent state (None to reset). For rnn it is the hidden
               tensor, for lstm the (h, c) tuple, for mpn a list of M matrices.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        model_type: str = "lstm",
        activation: str = "tanh",
        lambda_max: float = 0.99,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
        num_layers: int = 1,
        mpn_bias: bool = True,
        random_proj_dim: int | None = None,
    ):
        super().__init__()
        self.core = RecurrentCore(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            model_type=model_type,
            activation=activation,
            lambda_max=lambda_max,
            eta_init=eta_init,
            lambda_init=lambda_init,
            num_layers=num_layers,
            mpn_bias=mpn_bias,
            random_proj_dim=random_proj_dim,
        )
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

    def forward(self, x: torch.Tensor, state: torch.Tensor | tuple | list | None):
        out, state = self.core(x, state)
        out = self.postprocessor(out)
        return self.actor(out), self.critic(out), state
