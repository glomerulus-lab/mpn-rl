from typing import Any

import neurogym
import numpy as np
import numpy.typing as npt

# Steps to score, keyed on NeuroGym's own period schedule rather than a gt/input proxy.
# The default scored window is the "decision" period; the match-with- distractors task
# has no single decision period and is scored across its test windows (and the delays
# where it must withhold) instead.
RESPONSE_PERIODS: dict[str, list[str]] = {
    "ContextDecisionMaking-v0": ["decision"],
    "DelayComparison-v0": ["decision"],
    "DelayMatchSample-v0": ["decision"],
    "IntervalDiscrimination-v0": ["decision"],
    "MultiSensoryIntegration-v0": ["decision"],
    "PerceptualDecisionMaking-v0": ["decision"],
    "PerceptualDecisionMakingDelayResponse-v0": ["decision"],
    "ProbabilisticReasoning-v0": ["decision"],
    "GoNogo-v0": ["decision"],
    "DelayPairedAssociation-v0": ["decision"],
    "DelayMatchSampleDistractor1D-v0": [
        "delay1",
        "test1",
        "delay2",
        "test2",
        "delay3",
        "test3",
    ],
}


class MaskedSequenceSampler:
    """Rolls NeuroGym trials into fixed-length (batch, time) sequences plus a
    response-period mask.

    Mirrors neurogym.Dataset's batching but also reads each trial's period
    schedule (start_ind/end_ind) — which Dataset discards — to mark the steps
    where a response is expected. State resets per sequence, so the trailing
    partial trial is truncated rather than carried across batches.
    """

    def __init__(
        self,
        env_name: str,
        env_kwargs: dict[str, Any],
        batch_size: int,
        seq_len: int,
        seed: int,
    ):
        self._envs = [neurogym.make(env_name, **env_kwargs) for _ in range(batch_size)]
        for i, wrapped_env in enumerate(self._envs):
            wrapped_env.reset(seed=seed + i)
            # reset(seed=) seeds gymnasium's RNG, not NeuroGym's trial RNG — seed
            # that too so trial generation is reproducible.
            wrapped_env.unwrapped.seed(seed + i)
        self._seq_len = seq_len
        self._response_periods = RESPONSE_PERIODS[env_name]
        self._input_dim: int = self._envs[0].observation_space.shape[0]
        self._num_classes: int = self._envs[0].action_space.n

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def sample(
        self,
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
        batch = len(self._envs)
        inputs = np.zeros((batch, self._seq_len, self._input_dim), dtype=np.float32)
        targets = np.zeros((batch, self._seq_len), dtype=np.int64)
        mask = np.zeros((batch, self._seq_len), dtype=np.bool_)
        for i, wrapped_env in enumerate(self._envs):
            env = wrapped_env.unwrapped  # the trial schedule lives on the base env
            t = 0
            while t < self._seq_len:
                env.new_trial()
                observation, ground_truth = env.ob, env.gt
                trial_mask = np.zeros(len(ground_truth), dtype=np.bool_)
                for period in self._response_periods:
                    trial_mask[env.start_ind[period] : env.end_ind[period]] = True
                n = min(len(ground_truth), self._seq_len - t)
                inputs[i, t : t + n] = observation[:n]
                targets[i, t : t + n] = ground_truth[:n]
                mask[i, t : t + n] = trial_mask[:n]
                t += n
        return inputs, targets, mask
