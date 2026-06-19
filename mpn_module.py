"""
Standalone Multi-Plasticity Network (MPN) Module for Reinforcement Learning

This module implements a simplified MPN layer with Hebbian plasticity that can be used
with standard RL frameworks. It follows an RNN-like interface for easy integration with
Gym environments.

Key features:
- Hebbian synaptic modulation: M_t = λM_{t-1} + ηh_t x_t^T
- Multiplicative plasticity: output = activation(W*(M+1)*x + b)
- RNN-like interface: forward(x, state) -> (output, new_state)
- Minimal dependencies: PyTorch only

Reference: eLife-83035 - Multi-plasticity networks
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class RandomInputProjection(nn.Module):
    """
    Fixed random input projection: x_out = W_rand @ x_in + b_rand

    Implements the NeuroGym input expansion from eLife-83035 (Methods 5.1).
    For low-dimensional inputs, projects to a higher-dimensional space so that
    near-zero inputs still produce a non-trivial modulation-dependent response.

    W_rand and b_rand are initialized with Xavier uniform initialization and
    are fixed (not trained) during training — registered as buffers.

    Args:
        input_dim: Original input dimensionality (d')
        output_dim: Projected dimensionality (d, default 10 in the paper)
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()

        W = torch.empty(output_dim, input_dim)
        b = torch.empty(output_dim)

        nn.init.xavier_uniform_(W)
        # Xavier uniform bound for the bias
        bound = np.sqrt(6.0 / (input_dim + output_dim))
        nn.init.uniform_(b, -bound, bound)

        self.register_buffer('W_rand', W)
        self.register_buffer('b_rand', b)

        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.W_rand, self.b_rand)


class MPNLayer(nn.Module):
    """
    Multi-Plasticity Network Layer with Hebbian learning.

    This layer maintains a synaptic modulation matrix M that is updated according to
    Hebbian rules during the forward pass. The M matrix acts as fast, within-episode
    memory, while the weight matrix W is learned via backpropagation across episodes.

    Args:
        input_dim: Dimension of input features
        hidden_dim: Dimension of hidden layer (output of this layer)
        activation: Activation function ('relu', 'tanh', 'sigmoid', or 'linear')
        bias: Whether to use bias term
        freeze_plasticity: Disable Hebbian updates
        lambda_max: Maximum value for lambda clamping (default 0.99)

    Shape:
        - Input: (batch_size, input_dim)
        - State: (batch_size, hidden_dim, input_dim)  # M matrix
        - Output: (batch_size, hidden_dim)
        - New State: (batch_size, hidden_dim, input_dim)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'tanh',
        bias: bool = True,
        freeze_plasticity: bool = False,
        lambda_max: float = 0.95,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freeze_plasticity = freeze_plasticity
        self.lambda_max = lambda_max

        # Hebbian plasticity parameters (learnable via backprop)
        # Eta: Xavier-uniform init so it can start positive or negative (anti-Hebbian)
        bound = np.sqrt(3.0)  # Xavier uniform for a scalar ~ U[-sqrt(3), sqrt(3)]
        eta_val = np.random.uniform(-bound, bound) if eta_init is None else eta_init
        self.eta = nn.Parameter(torch.tensor(eta_val, dtype=torch.float32))
        self._lambda_raw = nn.Parameter(torch.tensor(lambda_init, dtype=torch.float32))

        # Long-term synaptic weights (trainable via backprop)
        # Shape: [hidden_dim, input_dim]
        self.W = nn.Parameter(torch.randn(hidden_dim, input_dim) * np.sqrt(2.0 / input_dim))

        # Bias term (trainable)
        if bias:
            self.b = nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_buffer('b', torch.zeros(hidden_dim))

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'linear':
            self.activation = nn.Identity()
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Initialize the synaptic modulation matrix M to zeros.

        Args:
            batch_size: Number of parallel sequences
            device: Device to create the state on

        Returns:
            Initial M matrix of shape (batch_size, hidden_dim, input_dim)
        """
        if device is None:
            device = self.W.device
            
        return torch.zeros(batch_size, self.hidden_dim, self.input_dim, device=device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Hebbian plasticity update.

        The forward pass computes:
        1. Pre-activation: y_tilde = b + (W * (M + 1)) @ x
        2. Activation: h = activation(y_tilde)
        3. Hebbian update: M_new = λ*M + η*h*x^T

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            state: Current M matrix of shape (batch_size, hidden_dim, input_dim).
                   If None, initializes to zeros.

        Returns:
            output: Hidden activations of shape (batch_size, hidden_dim)
            new_state: Updated M matrix of shape (batch_size, hidden_dim, input_dim)
        """
        batch_size = x.shape[0]

        # Initialize state if not provided
        if state is None:
            M = self.init_state(batch_size, device=x.device)
        else:
            M = state

        # When plasticity is frozen, M is always zero, so W_modulated = W
        # This optimization avoids unnecessary computation
        if self.freeze_plasticity:
            # Simplified computation: W @ x + b (M is always zero)
            # W shape: [hidden_dim, input_dim]
            # x shape: [batch_size, input_dim]
            # Result: [batch_size, hidden_dim]
            y_tilde = torch.nn.functional.linear(x, self.W, self.b)

            # Apply activation
            h = self.activation(y_tilde)

            # Return zeros for M (detached to save memory)
            M_new = torch.zeros_like(M).detach()
        else:
            # Compute modulated weights: W * (M + 1)
            # W shape: [hidden_dim, input_dim]
            # M shape: [batch_size, hidden_dim, input_dim]
            # Result: [batch_size, hidden_dim, input_dim]
            W_modulated = self.W.unsqueeze(0) * (M + 1.0)

            # Compute pre-activation: b + W_modulated @ x
            # x shape: [batch_size, input_dim, 1]
            # W_modulated @ x: [batch_size, hidden_dim, 1]
            # Result: [batch_size, hidden_dim]
            y_tilde = self.b.unsqueeze(0) + torch.bmm(W_modulated, x.unsqueeze(2)).squeeze(2)

            # Apply activation
            h = self.activation(y_tilde)

            # Hebbian update: M_new = λ*M + η*h*x^T
            # Clamp lambda in-place (like the reference) so the parameter itself
            # stays bounded — prevents gradients from blocking above lambda_max
            with torch.no_grad():
                self._lambda_raw.data.clamp_(0.0, self.lambda_max)
            M_new = self._lambda_raw * M + self.eta * torch.bmm(
                h.unsqueeze(2), x.unsqueeze(1)
            )

        return h, M_new


class MPN(nn.Module):
    """
    Complete Multi-Plasticity Network for RL.

    This is a simple 2-layer network:
    - Layer 1: MPN layer with Hebbian plasticity
    - Layer 2: Linear readout (returns hidden states for use by policy/value heads)

    The network maintains internal state (M matrix) that persists across time steps
    within an episode but resets between episodes.

    Args:
        input_dim: Dimension of observations
        hidden_dim: Dimension of hidden layer
        activation: Activation function for hidden layer
        freeze_plasticity: Disable Hebbian updates
        lambda_max: Maximum value for lambda clamping (default 0.99)

    Example:
        >>> mpn = MPN(input_dim=4, hidden_dim=64)
        >>> obs = torch.randn(32, 4)  # batch of 32 observations
        >>>
        >>> # Initialize episode
        >>> state = mpn.init_state(batch_size=32)
        >>>
        >>> # Step through episode
        >>> for t in range(episode_length):
        >>>     hidden, state = mpn(obs[t], state)
        >>>     # Use hidden for policy/value computation
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        activation: str = 'tanh',
        freeze_plasticity: bool = False,
        lambda_max: float = 0.99,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
        bias: bool = True,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.freeze_plasticity = freeze_plasticity

        # MPN layer
        self.mpn_layer = MPNLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            activation=activation,
            bias=bias,
            freeze_plasticity=freeze_plasticity,
            lambda_max=lambda_max,
            eta_init=eta_init,
            lambda_init=lambda_init,
        )

    def init_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """Initialize the M matrix for a new episode."""
        return self.mpn_layer.init_state(batch_size, device)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the MPN.

        Args:
            x: Observations of shape (batch_size, input_dim)
            state: Current M matrix. If None, initializes to zeros.

        Returns:
            hidden: Hidden representations of shape (batch_size, hidden_dim)
            new_state: Updated M matrix
        """
        return self.mpn_layer(x, state)


if __name__ == "__main__":
    # Simple test
    print("Testing MPN module...")

    # Create MPN
    mpn = MPN(input_dim=4, hidden_dim=8)

    # Test single step
    batch_size = 2
    x = torch.randn(batch_size, 4)
    state = mpn.init_state(batch_size)

    hidden, new_state = mpn(x, state)

    print(f"Input shape: {x.shape}")
    print(f"State shape: {state.shape}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"New state shape: {new_state.shape}")
    print(f"\nState changed: {not torch.allclose(state, new_state)}")

    # Test sequence
    print("\nTesting sequence of 5 steps:")
    state = mpn.init_state(batch_size)
    for t in range(5):
        x = torch.randn(batch_size, 4)
        hidden, state = mpn(x, state)
        print(f"Step {t}: M matrix mean = {state.mean().item():.4f}, std = {state.std().item():.4f}")

    print("\nMPN module test completed!")

