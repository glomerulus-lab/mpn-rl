import numpy as np
import pytest

from mpn_rl.supervised_data import MaskedSequenceSampler


def test_response_periods_raises_for_unmapped_env() -> None:
    with pytest.raises(KeyError):
        MaskedSequenceSampler("ReadySetGo-v0", {}, batch_size=4, seq_len=80, seed=0)


def test_sampler_mask_equals_gt_positive_for_label_task() -> None:
    # Label tasks emit gt 0 everywhere but the decision period, so the schedule-
    # derived mask should match gt>0 exactly (the reference impl's "label" rule).
    sampler = MaskedSequenceSampler(
        "PerceptualDecisionMaking-v0", {}, batch_size=4, seq_len=80, seed=0
    )
    _, targets, mask = sampler.sample()
    assert np.array_equal(mask, targets > 0)


def test_sampler_mask_equals_fixation_off_for_gonogo() -> None:
    # GoNogo's no-go response is action 0, so a gt>0 mask would drop it. The scored
    # window is instead "fixation cue (channel 0) off" (the reference's "no_fix").
    sampler = MaskedSequenceSampler("GoNogo-v0", {}, batch_size=4, seq_len=80, seed=0)
    inputs, _, mask = sampler.sample()
    assert np.array_equal(mask, inputs[..., 0] == 0)


def test_sampler_distractor_scores_tests_but_not_sample_period() -> None:
    # We have no response-period flag to check against, so we use the fixation cue
    # (channel 0). The scored windows are delay1..test3, and the cue is off
    # throughout them, so every scored step is a cue-off step. The sample window is
    # cue-off too but is not scored, so fewer steps are scored than are cue-off.
    sampler = MaskedSequenceSampler(
        "DelayMatchSampleDistractor1D-v0", {}, batch_size=4, seq_len=120, seed=0
    )
    inputs, _, mask = sampler.sample()
    fixation_off = inputs[..., 0] == 0
    assert (mask <= fixation_off).all()
    assert mask.sum() < fixation_off.sum()


def test_sampler_returns_expected_shapes() -> None:
    sampler = MaskedSequenceSampler("GoNogo-v0", {}, batch_size=4, seq_len=50, seed=0)
    inputs, targets, mask = sampler.sample()
    assert inputs.shape == (4, 50, sampler.input_dim)
    assert targets.shape == (4, 50)
    assert mask.shape == (4, 50)


def test_sampler_reproducible_from_seed() -> None:
    a = MaskedSequenceSampler(
        "GoNogo-v0", {}, batch_size=4, seq_len=50, seed=0
    ).sample()
    b = MaskedSequenceSampler(
        "GoNogo-v0", {}, batch_size=4, seq_len=50, seed=0
    ).sample()
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_sampler_seed_changes_sequences() -> None:
    a, _, _ = MaskedSequenceSampler(
        "GoNogo-v0", {}, batch_size=4, seq_len=50, seed=0
    ).sample()
    b, _, _ = MaskedSequenceSampler(
        "GoNogo-v0", {}, batch_size=4, seq_len=50, seed=1
    ).sample()
    assert not np.array_equal(a, b)
