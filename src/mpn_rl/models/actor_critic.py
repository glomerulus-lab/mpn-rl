"""Actor-critic network for episode-based A2C training."""

import torch
import torch.nn as nn

from mpn_rl.nn.mpn import MPN


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
        model_type: str = "lstm",
        activation: str = "tanh",
        lambda_max: float = 0.99,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
        num_layers: int = 1,
        mpn_bias: bool = True,
        lstm_forget_bias: float = 0.0,
    ):
        super().__init__()
        self.model_type = model_type
        self.num_layers = num_layers

        if model_type == "rnn":
            self.core = nn.RNN(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
        elif model_type == "lstm":
            self.core = nn.LSTM(
                input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
            if lstm_forget_bias != 0.0:
                for name, param in self.core.named_parameters():
                    if "bias" in name:
                        n = param.size(0)
                        param.data[n // 4 : n // 2].fill_(lstm_forget_bias)
        elif model_type in ("mpn", "mpn-frozen"):
            freeze = model_type == "mpn-frozen"
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
            raise ValueError(f"Unknown model_type: {model_type!r}")

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
        if self.model_type in ("rnn", "lstm"):
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
