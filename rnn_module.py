"""
Custom RNNModule for TorchRL based on the GRUModule implementation.
This wraps PyTorch's nn.RNN to make it compatible with TensorDict.
"""

import torch
from tensordict import TensorDictBase, unravel_key_list
from tensordict.base import NO_DEFAULT
from tensordict.nn import TensorDictModuleBase as ModuleBase
from tensordict.nn import dispatch
from tensordict.utils import expand_as_right, prod
from torch import nn
from torchrl.data.tensor_specs import Unbounded


class RNNModule(ModuleBase):
    """An embedder for an RNN module.

    This class adds the following functionality to :class:`torch.nn.RNN`:

    - Compatibility with TensorDict: the hidden states are reshaped to match
      the tensordict batch size.
    - Optional multi-step execution: with torch.nn, one has to choose between
      :class:`torch.nn.RNNCell` and :class:`torch.nn.RNN`, the former being
      compatible with single step inputs and the latter being compatible with
      multi-step. This class enables both usages.

    After construction, the module is *not* set in recurrent mode, ie. it will
    expect single steps inputs.

    If in recurrent mode, it is expected that the last dimension of the tensordict
    marks the number of steps.

    Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. Default: 1
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'. Default: 'tanh'
        bias: If ``False``, then the layer does not use bias weights. Default: ``True``
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer. Default: 0

    Keyword Args:
        in_key (str or tuple of str): the input key of the module.
        in_keys (list of str): a pair of strings corresponding to the input value and recurrent entry.
        out_key (str or tuple of str): the output key of the module.
        out_keys (list of str): a pair of strings corresponding to the output value and hidden state.
        device (torch.device or compatible): the device of the module.
        default_recurrent_mode (bool, optional): if provided, the recurrent mode.
            Defaults to ``False``.

    """

    DEFAULT_IN_KEYS = ["recurrent_state"]
    DEFAULT_OUT_KEYS = [("next", "recurrent_state")]

    def __init__(
        self,
        input_size: int = None,
        hidden_size: int = None,
        num_layers: int = 1,
        nonlinearity: str = 'tanh',
        bias: bool = True,
        batch_first=True,
        dropout=0,
        bidirectional=False,
        *,
        in_key=None,
        in_keys=None,
        out_key=None,
        out_keys=None,
        device=None,
        rnn=None,
        default_recurrent_mode: bool | None = None,
    ):
        super().__init__()
        if rnn is not None:
            if not rnn.batch_first:
                raise ValueError("The input rnn must have batch_first=True.")
            if rnn.bidirectional:
                raise ValueError("The input rnn cannot be bidirectional.")
            if input_size is not None or hidden_size is not None:
                raise ValueError(
                    "An RNN instance cannot be passed along with class argument."
                )
        else:
            if not batch_first:
                raise ValueError("The input rnn must have batch_first=True.")
            if bidirectional:
                raise ValueError("The input rnn cannot be bidirectional.")

            rnn = nn.RNN(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                nonlinearity=nonlinearity,
                bias=bias,
                dropout=dropout,
                device=device,
                batch_first=True,
                bidirectional=False,
            )

        if not ((in_key is None) or (in_keys is None)):
            raise ValueError(
                f"Either in_keys or in_key must be specified but not both or none. Got {in_keys} and {in_key} respectively."
            )
        elif in_key:
            in_keys = [in_key, *self.DEFAULT_IN_KEYS]

        if not ((out_key is None) or (out_keys is None)):
            raise ValueError(
                f"Either out_keys or out_key must be specified but not both or none. Got {out_keys} and {out_key} respectively."
            )
        elif out_key:
            out_keys = [out_key, *self.DEFAULT_OUT_KEYS]

        in_keys = unravel_key_list(in_keys)
        out_keys = unravel_key_list(out_keys)
        if not isinstance(in_keys, (tuple, list)) or (
            len(in_keys) != 2 and not (len(in_keys) == 3 and in_keys[-1] == "is_init")
        ):
            raise ValueError(
                f"RNNModule expects 2 inputs: a value and a hidden state (and potentially an 'is_init' marker). Got in_keys {in_keys} instead."
            )
        if not isinstance(out_keys, (tuple, list)) or len(out_keys) != 2:
            raise ValueError(
                f"RNNModule expects 2 outputs: a value and a hidden state. Got out_keys {out_keys} instead."
            )
        self.rnn = rnn
        if "is_init" not in in_keys:
            in_keys = in_keys + ["is_init"]
        self.in_keys = in_keys
        self.out_keys = out_keys
        self._recurrent_mode = default_recurrent_mode

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment."""
        from torchrl.envs import TensorDictPrimer

        def make_tuple(key):
            if isinstance(key, tuple):
                return key
            return (key,)

        out_key1 = make_tuple(self.out_keys[1])
        in_key1 = make_tuple(self.in_keys[1])
        if out_key1 != ("next", *in_key1):
            raise RuntimeError(
                "make_tensordict_primer is supposed to work with in_keys/out_keys that "
                "have compatible names, ie. the out_keys should be named after ('next', <in_key>). Got "
                f"in_keys={self.in_keys} and out_keys={self.out_keys} instead."
            )
        return TensorDictPrimer(
            {
                in_key1: Unbounded(shape=(self.rnn.num_layers, self.rnn.hidden_size)),
            },
            expand_specs=True,
        )

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
    def forward(self, tensordict: TensorDictBase):
        from torchrl.objectives.value.functional import (
            _inv_pad_sequence,
            _split_and_pad_sequence,
        )

        # we want to get an error if the value input is missing, but not the hidden states
        defaults = [NO_DEFAULT, None]
        shape = tensordict.shape
        tensordict_shaped = tensordict
        if self.recurrent_mode:
            # if less than 2 dims, unsqueeze
            ndim = tensordict_shaped.get(self.in_keys[0]).ndim
            while ndim < 3:
                tensordict_shaped = tensordict_shaped.unsqueeze(0)
                ndim += 1
            if ndim > 3:
                dims_to_flatten = ndim - 3
                # we assume that the tensordict can be flattened like this
                nelts = prod(tensordict_shaped.shape[: dims_to_flatten + 1])
                tensordict_shaped = tensordict_shaped.apply(
                    lambda value: value.flatten(0, dims_to_flatten),
                    batch_size=[nelts, tensordict_shaped.shape[-1]],
                )
        else:
            tensordict_shaped = tensordict.reshape(-1).unsqueeze(-1)

        is_init = tensordict_shaped["is_init"].squeeze(-1)
        splits = None
        if self.recurrent_mode and is_init[..., 1:].any():
            from torchrl.objectives.value.utils import _get_num_per_traj_init

            splits = _get_num_per_traj_init(is_init)
            tensordict_shaped_shape = tensordict_shaped.shape
            tensordict_shaped = _split_and_pad_sequence(
                tensordict_shaped.select(*self.in_keys, strict=False), splits
            )
            is_init = tensordict_shaped["is_init"].squeeze(-1)

        value, hidden = (
            tensordict_shaped.get(key, default)
            for key, default in zip(self.in_keys, defaults)
        )
        batch, steps = value.shape[:2]
        device = value.device
        dtype = value.dtype

        if is_init.any() and hidden is not None:
            is_init_expand = expand_as_right(is_init, hidden)
            hidden = torch.where(is_init_expand, 0, hidden)
        val, hidden = self._rnn(value, batch, steps, device, dtype, hidden)
        tensordict_shaped.set(self.out_keys[0], val)
        tensordict_shaped.set(self.out_keys[1], hidden)
        if splits is not None:
            # let's recover our original shape
            tensordict_shaped = _inv_pad_sequence(tensordict_shaped, splits).reshape(
                tensordict_shaped_shape
            )

        if shape != tensordict_shaped.shape or tensordict_shaped is not tensordict:
            tensordict.update(tensordict_shaped.reshape(shape))
        return tensordict

    def _rnn(
        self,
        input: torch.Tensor,
        batch,
        steps,
        device,
        dtype,
        hidden_in: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Internal RNN forward pass.

        PyTorch's nn.RNN can efficiently process multi-step sequences,
        so we support both steps=1 (rollout) and steps>1 (training with sequences).

        Args:
            input: [batch, steps, input_size]
            batch: batch size
            steps: number of time steps in the sequence
            device: device
            dtype: data type
            hidden_in: [batch, steps, num_layers, hidden_size] or None

        Returns:
            output: [batch, steps, hidden_size]
            hidden_out: [batch, steps, num_layers, hidden_size]
        """
        if hidden_in is None:
            shape = (batch, steps)
            hidden_in = torch.zeros(
                *shape,
                self.rnn.num_layers,
                self.rnn.hidden_size,
                device=device,
                dtype=dtype,
            )

        # we only need the first hidden state
        _hidden_in = hidden_in[:, 0]
        hidden = _hidden_in.transpose(-3, -2).contiguous()

        # PyTorch's nn.RNN processes all steps at once
        # input: [batch, steps, input_size]
        # hidden: [num_layers, batch, hidden_size]
        # output y: [batch, steps, hidden_size]
        # output hidden: [num_layers, batch, hidden_size] (state after last step)
        y, hidden = self.rnn(input, hidden)

        # dim 0 in hidden is num_layers, but that will conflict with tensordict
        hidden = hidden.transpose(0, 1)

        # we pad the hidden states with zero to make tensordict happy
        # Following LSTM pattern: zeros for all steps except last
        hidden = torch.stack(
            [torch.zeros_like(hidden) for _ in range(steps - 1)] + [hidden],
            1,
        )
        out = [y, hidden]
        return tuple(out)
