"""
TorchRL-compatible MPN Module

This module provides a TorchRL-compatible wrapper for the Multi-Plasticity Network (MPN) layer.
It follows the same interface as LSTMModule, making it a drop-in replacement for recurrent
modules in TorchRL pipelines.

Key features:
- TensorDict integration with automatic state management
- Compatible with TensorDictSequential pipelines
- Primer transform for automatic state initialization
- Support for both single-step and recurrent modes

Example:
    >>> from mpn_torchrl_module import MPNModule
    >>> from torchrl.envs import GymEnv
    >>>
    >>> # Create environment
    >>> env = GymEnv("CartPole-v1")
    >>>
    >>> # Create MPN module
    >>> mpn = MPNModule(
    ...     input_size=4,
    ...     hidden_size=64,
    ...     in_key="observation",
    ...     out_key="embed"
    ... )
    >>>
    >>> # Add primer to environment
    >>> env.append_transform(mpn.make_tensordict_primer())
    >>>
    >>> # Use in policy
    >>> from tensordict.nn import TensorDictSequential
    >>> from torchrl.modules import MLP, QValueModule
    >>>
    >>> policy = TensorDictSequential(
    ...     mpn,
    ...     MLP(in_features=64, out_features=2),
    ...     QValueModule(spec=env.action_spec)
    ... )
"""

from math import prod
from typing import Optional

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase, unravel_key_list
from tensordict.base import NO_DEFAULT
from tensordict.nn import TensorDictModuleBase as ModuleBase, dispatch
from torchrl.data.tensor_specs import Unbounded
from torchrl.envs.transforms import TensorDictPrimer

from mpn_module import MPNLayer


class MPNModule(ModuleBase):
    """
    TorchRL-compatible Multi-Plasticity Network module.

    This module wraps the MPNLayer to provide seamless integration with TorchRL's
    TensorDict-based infrastructure. It maintains a recurrent state (M matrix) that
    persists across time steps within an episode.

    Optionally includes pre-MPN linear layers that project the input through
    a sequence of Linear + tanh layers before the MPN layer.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden layer (output)
        activation: Activation function ('relu', 'tanh', 'sigmoid', 'linear')
        bias: Whether to use bias term (default: True)
        freeze_plasticity: Disable Hebbian updates (default: False)
        lambda_max: Maximum value for lambda clamping (default 0.99)
        num_pre_layers: Number of Linear+tanh layers before MPN (default: 0)
        in_key: Input key in TensorDict (default: "observation")
        out_key: Output key in TensorDict (default: "embed")
        device: Device to place module on (default: None)
    """

    DEFAULT_IN_KEYS = ["recurrent_state_m"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state_m")]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        activation: str = "tanh",
        bias: bool = True,
        freeze_plasticity: bool = False,
        lambda_max: float = 0.99,
        eta_init: float = 0.01,
        lambda_init: float = 0.99,
        num_pre_layers: int = 0,
        in_key: Optional[str] = None,
        in_keys: Optional[list] = None,
        out_key: Optional[str] = None,
        out_keys: Optional[list] = None,
        device: Optional[torch.device] = None,
        default_recurrent_mode: bool | None = None,
    ):
        super().__init__()

        # Handle in_keys validation (similar to LSTMModule)
        if not ((in_key is None) ^ (in_keys is None)):
            # Default to "observation" if neither provided
            if in_key is None and in_keys is None:
                in_key = "observation"
                in_keys = [in_key, *self.DEFAULT_IN_KEYS]
            else:
                raise ValueError(
                    f"Either in_keys or in_key must be specified but not both. "
                    f"Got in_keys={in_keys} and in_key={in_key}"
                )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        # Handle out_keys validation
        if not ((out_key is None) ^ (out_keys is None)):
            if out_key is None and out_keys is None:
                out_key = "embed"
                out_keys = [out_key, *self.DEFAULT_OUT_KEYS]
            else:
                raise ValueError(
                    f"Either out_keys or out_key must be specified but not both. "
                    f"Got out_keys={out_keys} and out_key={out_key}"
                )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        # Unravel keys
        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)

        # Validate key counts
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 2 and not (len(in_keys) == 3 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"MPNModule expects 2 inputs: a value and recurrent state "
                f"(and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 2:
            raise ValueError(
                f"MPNModule expects 2 outputs: a value and recurrent state. "
                f"Got out_keys {out_keys} instead."
            )

        # Add is_init if not present
        if "is_init" not in in_keys:
            in_keys = list(in_keys) + ["is_init"]

        self.in_keys = in_keys
        self.out_keys = out_keys

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.freeze_plasticity = freeze_plasticity

        # Build pre-MPN linear layers: Linear + tanh
        self.pre_layers = nn.ModuleList()
        pre_input_dim = input_size
        for _ in range(num_pre_layers):
            self.pre_layers.append(nn.Linear(pre_input_dim, hidden_size))
            pre_input_dim = hidden_size

        # The MPN layer input_dim is hidden_size if we have pre-layers, else input_size
        mpn_input_dim = hidden_size if num_pre_layers > 0 else input_size

        # Create MPN layer (eta and lambda are trainable parameters internally)
        self.mpn_layer = MPNLayer(
            input_dim=mpn_input_dim,
            hidden_dim=hidden_size,
            activation=activation,
            bias=bias,
            freeze_plasticity=freeze_plasticity,
            lambda_max=lambda_max,
            eta_init=eta_init,
            lambda_init=lambda_init,
        )

        # Store the actual MPN input dim for state shape
        self._mpn_input_dim = mpn_input_dim
        self._recurrent_mode = default_recurrent_mode

        if device is not None:
            self.to(device)

    @property
    def recurrent_mode(self):
        from torchrl.modules.tensordict_module.rnn import recurrent_mode

        rm = recurrent_mode()
        if rm is None:
            return bool(self._recurrent_mode)
        return rm

    @recurrent_mode.setter
    def recurrent_mode(self, value):
        raise RuntimeError(
            "recurrent_mode cannot be changed in-place. Please use the set_recurrent_mode context manager."
        )

    @dispatch
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Forward pass through the MPN layer using TensorDict.

        Mirrors TorchRL's LSTMModule pattern exactly:
        - Non-recurrent mode (rollout): reshape to [batch, 1, features], single step.
        - Recurrent mode (training): keep [batch, T, features], full sequence with BPTT.

        Args:
            tensordict: Input TensorDict containing observations and optionally
                       recurrent states and is_init flags

        Returns:
            Updated TensorDict with hidden activations and new recurrent state
        """
        from tensordict.utils import expand_as_right
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )

        defaults = [NO_DEFAULT, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict

        if self.recurrent_mode:
            # Ensure at least 3 dims: [batch, time, feature]
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped.get("is_init", None)
        if is_init is not None:
            is_init = is_init.squeeze(-1)

        # Handle episode boundaries within a sequence (multi-trajectory batches)
        splits = None
        if self.recurrent_mode and is_init is not None and is_init[..., 1:].any():
            from torchrl.objectives.value.utils import _get_num_per_traj_init

            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped.get("is_init", None)
            if is_init is not None:
                is_init = is_init.squeeze(-1)

        value, recurrent_state_m = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys[:2], defaults)
        )

        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype

        # In non-recurrent mode, reset state at episode boundaries (mirrors LSTM line 734-741)
        if (
            not self.recurrent_mode
            and recurrent_state_m is not None
            and is_init is not None
        ):
            is_init_expand = expand_as_right(is_init, recurrent_state_m)
            recurrent_state_m = torch.where(is_init_expand, 0, recurrent_state_m)

        val, new_state_m = self._mpn(
            value, batch, steps, device, dtype, recurrent_state_m
        )

        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], new_state_m)

        if splits is not None:
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))

        return tensordict

    def _apply_pre_layers(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pre-MPN linear layers with tanh activation."""
        for linear in self.pre_layers:
            x = torch.tanh(linear(x))
        return x

    def _mpn(
        self,
        input: torch.Tensor,
        batch: int,
        steps: int,
        device: torch.device,
        dtype: torch.dtype,
        state_m_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Internal MPN forward - mirrors LSTM._lstm pattern exactly.

        In non-recurrent mode (rollout): steps == 1, processes a single step.
        In recurrent mode (training): steps == T, unrolls full sequence with BPTT.
        Gradients flow through the Hebbian M update chain, enabling eta and lambda
        to receive meaningful gradient signals during sequence training.

        Args:
            input: [batch, steps, input_size]
            batch: batch size
            steps: number of time steps in the sequence
            device: device
            dtype: data type
            state_m_in: [batch, steps, hidden_size, mpn_input_dim] or None

        Returns:
            output: [batch, steps, hidden_size]
            state_m_out: [batch, steps, hidden_size, mpn_input_dim]
                Zero-padded at positions 0..T-2; actual final M at position T-1.
                (Mirrors LSTM convention — only state[:, 0] is used as starting
                state when replaying the next sequence segment.)
        """
        if not self.recurrent_mode and steps != 1:
            raise ValueError(
                f"MPNModule expects steps=1 in non-recurrent mode, got steps={steps}. "
                "Use set_recurrent_mode(True) during training with sequence replay."
            )

        # Initialize state if not provided
        if state_m_in is None:
            state_m_in = torch.zeros(
                batch,
                steps,
                self.hidden_size,
                self._mpn_input_dim,
                device=device,
                dtype=dtype,
            )

        # Only the first step's stored state is used as the starting point.
        # In recurrent mode, M_1..M_T are recomputed in this forward pass,
        # giving gradients a path back to eta and lambda via BPTT.
        _state_m = state_m_in[:, 0]  # [batch, hidden_size, mpn_input_dim]

        if self.freeze_plasticity:
            outputs = []
            for t in range(steps):
                _input_t = self._apply_pre_layers(input[:, t])
                hidden_t, _ = self.mpn_layer(_input_t, _state_m)
                outputs.append(hidden_t)

            output = torch.stack(outputs, dim=1)
            state_m_out = torch.zeros(
                batch,
                steps,
                self.hidden_size,
                self._mpn_input_dim,
                device=device,
                dtype=dtype,
            ).detach()
        else:
            outputs = []
            for t in range(steps):
                _input_t = self._apply_pre_layers(input[:, t])
                hidden_t, _state_m = self.mpn_layer(_input_t, _state_m)
                outputs.append(hidden_t)

            output = torch.stack(outputs, dim=1)

            # Zero-pad intermediate positions — mirrors LSTM convention.
            # The stored final state is used as state[:, 0] for the next segment.
            state_m_out = torch.stack(
                [torch.zeros_like(_state_m) for _ in range(steps - 1)] + [_state_m],
                dim=1,
            )

        return output, state_m_out

    def make_tensordict_primer(self) -> TensorDictPrimer:
        """
        Create a TensorDictPrimer transform for automatic state initialization.

        This primer should be added to the environment transforms to ensure that
        the M matrix state is automatically initialized and tracked during rollouts.

        Returns:
            TensorDictPrimer that initializes recurrent states
        """

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        # Validate naming convention (out_key should be ("next", in_key))
        out_key = make_tuple(self.out_keys[1])
        in_key = make_tuple(self.in_keys[1])
        if out_key != ("next", *in_key):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )

        # Create primer with M matrix spec — uses mpn_input_dim (after pre-layers)
        return TensorDictPrimer(
            {
                in_key: Unbounded(shape=(self.hidden_size, self._mpn_input_dim)),
            },
            expand_specs=True,
        )


if __name__ == "__main__":
    print("Testing MPNModule with TorchRL...")

    # Test 1: Basic forward pass
    print("\n" + "=" * 50)
    print("Test 1: Basic forward pass")
    print("=" * 50)

    mpn = MPNModule(input_size=4, hidden_size=8, in_key="observation", out_key="embed")

    # Create sample tensordict
    batch_size = 3
    td = TensorDict(
        {
            "observation": torch.randn(batch_size, 4),
            "is_init": torch.tensor([True, False, True]),  # Reset first and third
        },
        batch_size=[batch_size],
    )

    print(f"Input keys: {list(td.keys())}")

    # Forward pass (state auto-initialized)
    td = mpn(td)

    print(f"Output keys: {list(td.keys())}")
    print(f"Hidden shape: {td['embed'].shape}")
    print(f"State shape: {td['next', 'recurrent_state_m'].shape}")
    print(f"MPN in_keys: {mpn.in_keys}")
    print(f"MPN out_keys: {mpn.out_keys}")

    # Test 2: State persistence
    print("\n" + "=" * 50)
    print("Test 2: State persistence across steps")
    print("=" * 50)

    # Copy state to input for next step
    td["recurrent_state_m"] = td["next", "recurrent_state_m"].clone()
    td["observation"] = torch.randn(batch_size, 4)
    td["is_init"] = torch.tensor([False, False, False])  # No resets

    old_state = td["recurrent_state_m"].clone()
    td = mpn(td)
    new_state = td["next", "recurrent_state_m"]

    print(f"State changed: {not torch.allclose(old_state, new_state)}")
    print(f"State mean: {new_state.mean().item():.4f}")

    # Test 3: Primer
    print("\n" + "=" * 50)
    print("Test 3: TensorDictPrimer")
    print("=" * 50)

    primer = mpn.make_tensordict_primer()
    print(f"Primer type: {type(primer)}")
    print(f"Primer keys: {list(primer.primers.keys())}")
    print("Skipping primer test - will test with actual environment")

    # Test 4: Use in sequential pipeline
    print("\n" + "=" * 50)
    print("Test 4: TensorDictSequential pipeline")
    print("=" * 50)

    from tensordict.nn import TensorDictModule, TensorDictSequential

    mpn_module = MPNModule(
        input_size=4, hidden_size=64, in_key="observation", out_key="embed"
    )

    # MLP head
    mlp = nn.Linear(64, 2)
    mlp_module = TensorDictModule(mlp, in_keys=["embed"], out_keys=["action_value"])

    # Create sequential
    policy = TensorDictSequential(mpn_module, mlp_module)

    # Test forward
    td_test = TensorDict(
        {"observation": torch.randn(5, 4), "is_init": torch.zeros(5, dtype=torch.bool)},
        batch_size=[5],
    )

    td_test = policy(td_test)
    print(f"Policy output keys: {list(td_test.keys())}")
    print(f"Action values shape: {td_test['action_value'].shape}")

    # Test 5: Episode simulation
    print("\n" + "=" * 50)
    print("Test 5: Multi-step episode simulation")
    print("=" * 50)

    mpn_sim = MPNModule(input_size=4, hidden_size=8)
    batch_size = 2
    num_steps = 5

    # Initialize episode
    td_sim = TensorDict(
        {
            "observation": torch.randn(batch_size, 4),
            "is_init": torch.ones(batch_size, dtype=torch.bool),  # Episode start
        },
        batch_size=[batch_size],
    )

    print("Step-by-step M matrix evolution:")
    for step in range(num_steps):
        td_sim = mpn_sim(td_sim)
        state_mean = td_sim["next", "recurrent_state_m"].mean().item()
        state_std = td_sim["next", "recurrent_state_m"].std().item()
        print(f"  Step {step}: M mean={state_mean:.4f}, std={state_std:.4f}")

        # Prepare next step
        if step < num_steps - 1:
            td_sim["recurrent_state_m"] = td_sim["next", "recurrent_state_m"].clone()
            td_sim["observation"] = torch.randn(batch_size, 4)
            td_sim["is_init"] = torch.zeros(batch_size, dtype=torch.bool)

    print("\nMPNModule tests completed successfully!")
