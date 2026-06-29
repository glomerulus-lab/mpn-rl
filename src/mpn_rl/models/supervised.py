"""Recurrent classifier for supervised NeuroGym training."""

import torch
import torch.nn as nn

from mpn_rl.nn.recurrent_core import RecurrentCore, RecurrentState


class SupervisedNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
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
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward(x) → logits

        x:      (batch, time, input_dim)
        logits: (batch, time, output_dim) — unnormalized, for cross-entropy

        State is reset at the start of the sequence and rolled across every
        timestep (rnn: hidden tensor, lstm: (h, c) tuple, mpn: list of M
        matrices). The model owns its own unrolling; callers pass whole
        sequences, not single steps.
        """
        state: RecurrentState = None
        hidden = []
        for t in range(x.shape[1]):
            out, state = self.core(x[:, t], state)
            hidden.append(out)
        logits: torch.Tensor = self.readout(torch.stack(hidden, dim=1))
        return logits
