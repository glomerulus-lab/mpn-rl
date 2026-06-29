"""Recurrent core shared by the A2C and supervised models."""

import math

import torch
import torch.nn as nn

from mpn_rl.nn.mpn import MPN, RandomInputProjection

RecurrentState = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor] | None
)


class RecurrentCore(nn.Module):
    """Optional fixed random input projection feeding an RNN / LSTM / MPN core.

    forward(x, state) → (hidden, new_state)
        x:      (batch, input_dim)
        hidden: (batch, hidden_dim)
        state:  core recurrent state (None to reset). For rnn it is the hidden
                tensor, for lstm the (h, c) tuple, for mpn a list of M matrices.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        model_type: str = "lstm",
        activation: str = "tanh",
        lambda_max: float = 0.99,
        eta_init: float | None = 0.01,
        lambda_init: float = 0.99,
        num_layers: int = 1,
        mpn_bias: bool = True,
        random_proj_dim: int | None = None,
        input_noise_scale: float = 0.0,
    ):
        super().__init__()
        self.model_type = model_type
        self.input_noise_scale = input_noise_scale

        if random_proj_dim is None:
            self.input_proj = None
            core_input_dim = input_dim
        else:
            self.input_proj = RandomInputProjection(input_dim, random_proj_dim)
            core_input_dim = random_proj_dim
        self.core_input_dim = core_input_dim

        self.core: nn.Module
        if model_type == "rnn":
            self.core = nn.RNN(
                core_input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
        elif model_type == "lstm":
            self.core = nn.LSTM(
                core_input_dim, hidden_dim, num_layers=num_layers, batch_first=True
            )
        elif model_type in ("mpn", "mpn-frozen"):
            self.core = MPN(
                core_input_dim,
                hidden_dim,
                num_layers=num_layers,
                activation=activation,
                lambda_max=lambda_max,
                eta_init=eta_init,
                lambda_init=lambda_init,
                freeze_plasticity=(model_type == "mpn-frozen"),
                bias=mpn_bias,
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

    def forward(
        self, x: torch.Tensor, state: RecurrentState
    ) -> tuple[torch.Tensor, RecurrentState]:
        if self.input_proj is not None:
            x = self.input_proj(x)
        if self.input_noise_scale > 0:
            # Add noise to the input, matching the reference implementation:
            # https://github.com/kaitken17/mpn/blob/6147f2b/networks.py#L334-L339
            std = self.input_noise_scale / math.sqrt(self.core_input_dim)
            x = x + std * torch.randn_like(x)
        if self.model_type in ("rnn", "lstm"):
            out, state = self.core(x.unsqueeze(1), state)  # (batch, 1, hidden)
            out = out.squeeze(1)  # (batch, hidden)
        else:  # MPN / MPN-frozen — state is a list of per-layer M matrices
            out, state = self.core(x, state)
        return out, state
