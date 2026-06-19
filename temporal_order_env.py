"""
Temporal Order Judgement Environment

The agent observes two stimuli (S1, S2) appearing at random distinct steps
within an n-step window and must report which appeared first at a response step.

Trial structure (n_steps=5, 7 steps per trial):
    t = 0..4  : stimulus window — S1 and S2 each appear once at a random step
    t = 5     : blank
    t = 6     : response cue — reward given based on action

Observation: [s1, s2, cue]  (3-dimensional)
Actions:     0 = S1 appeared first,  1 = S2 appeared first
Reward:      +1 correct at response step, -1 wrong, 0 everywhere else

Episode structure: one trial per episode — the episode terminates (terminated=True)
at the response step. The environment resets for the next episode. This ensures
each episode is a self-contained sequence of length trial_len, enabling clean
sequence-aligned replay buffer sampling.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TemporalOrderEnv(gym.Env):
    """
    Temporal order judgement task (continuous multi-trial episodes).

    Args:
        n_steps: Number of steps in the stimulus window (default 5).
                 Each trial is n_steps + 2 steps long.
    """

    metadata = {"render_modes": []}

    def __init__(self, n_steps: int = 5):
        super().__init__()
        assert n_steps >= 2, "n_steps must be >= 2 to fit both stimuli"
        self.n_steps = n_steps
        self.trial_len = n_steps + 2  # stimulus window + blank + response

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # Exposed for NeuroGymInfoWrapper compatibility
        self.trial = {"ground_truth": 0}
        self._t_in_trial = 0
        self._s1_step = 0
        self._s2_step = 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _new_trial(self):
        """Pre-build the full observation and ground-truth arrays for one trial.

        All RNG calls happen here so that seeding is fully deterministic —
        stepping never touches the RNG.
        """
        self._t_in_trial = 0

        # Sample S1 and S2 positions using the seeded RNG
        positions = self.np_random.choice(self.n_steps, size=2, replace=False)
        self._s1_step = int(positions[0])
        self._s2_step = int(positions[1])
        gt = 0 if self._s1_step < self._s2_step else 1

        # Pre-build observation array: shape (trial_len, 3)
        self._obs_array = np.zeros((self.trial_len, 3), dtype=np.float32)
        self._obs_array[self._s1_step, 0] = 1.0  # S1
        self._obs_array[self._s2_step, 1] = 1.0  # S2
        self._obs_array[self.trial_len - 1, 2] = 1.0  # response cue

        # Pre-build ground-truth array: gt only meaningful at response step
        self._gt_array = np.zeros(self.trial_len, dtype=np.int64)
        self._gt_array[self.trial_len - 1] = gt

        self.trial = {"ground_truth": gt}
        return gt

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        gt = self._new_trial()
        obs = self._obs_array[self._t_in_trial].copy()
        return obs, {"gt": gt, "new_trial": True}

    def step(self, action):
        is_response = self._t_in_trial == self.trial_len - 1

        if is_response:
            gt = self.trial["ground_truth"]
            reward = 1.0 if int(action) == gt else -1.0
            # Episode terminates — one trial per episode
            obs = np.zeros(3, dtype=np.float32)
            info = {"gt": gt, "new_trial": True}
            return obs, reward, True, False, info

        reward = 0.0
        self._t_in_trial += 1
        obs = self._obs_array[self._t_in_trial].copy()
        info = {"gt": self.trial["ground_truth"], "new_trial": False}
        return obs, reward, False, False, info


# ------------------------------------------------------------------
# Registration
# ------------------------------------------------------------------

gym.register(
    id="TemporalOrder-v0",
    entry_point="temporal_order_env:TemporalOrderEnv",
)

gym.register(
    id="TemporalOrder10-v0",
    entry_point="temporal_order_env:TemporalOrderEnv",
    kwargs={"n_steps": 10},
)

gym.register(
    id="TemporalOrder20-v0",
    entry_point="temporal_order_env:TemporalOrderEnv",
    kwargs={"n_steps": 20},
)


class TemporalOrderRandEnv(gym.Env):
    """
    Temporal order judgement with random binary input vectors (paper-style encoding).

    Instead of one-hot inputs [S1=[1,0,0], S2=[0,1,0], GO=[0,0,1]], uses fixed
    random binary vectors of dimension obs_dim with ~50% density. The GO vector
    overlaps with the S1/S2 channels, so the MPN's M-matrix is readable at the
    response step.

    This matches the encoding in Aitken & Mihalas (eLife 2023) Figure 2b, where
    each input type is a normalized random binary vector.

    Args:
        n_steps:     Number of steps in the stimulus window (default 5).
        obs_dim:     Observation dimension (default 16).
        vec_density: Fraction of active bits per vector (default 0.5).
        vec_seed:    Seed for generating the fixed input vectors (default 0).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_steps: int = 5,
        obs_dim: int = 16,
        vec_density: float = 0.5,
        vec_seed: int = 0,
    ):
        super().__init__()
        assert n_steps >= 2, "n_steps must be >= 2 to fit both stimuli"
        self.n_steps = n_steps
        self.trial_len = n_steps + 2
        self.obs_dim = obs_dim

        # Generate fixed random binary vectors once (same across all episodes)
        rng = np.random.default_rng(vec_seed)
        n_active = max(1, int(obs_dim * vec_density))

        def _make_vec():
            v = np.zeros(obs_dim, dtype=np.float32)
            v[rng.choice(obs_dim, size=n_active, replace=False)] = 1.0
            return v

        self.v_s1 = _make_vec()
        self.v_s2 = _make_vec()
        self.v_go = _make_vec()
        self.v_blank = np.zeros(obs_dim, dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        self.trial = {"ground_truth": 0}
        self._t_in_trial = 0
        self._s1_step = 0
        self._s2_step = 1

    def _new_trial(self):
        self._t_in_trial = 0

        positions = self.np_random.choice(self.n_steps, size=2, replace=False)
        self._s1_step = int(positions[0])
        self._s2_step = int(positions[1])
        gt = 0 if self._s1_step < self._s2_step else 1

        self._obs_array = np.zeros((self.trial_len, self.obs_dim), dtype=np.float32)
        for t in range(self.trial_len):
            if t == self._s1_step:
                self._obs_array[t] = self.v_s1
            elif t == self._s2_step:
                self._obs_array[t] = self.v_s2
            elif t == self.trial_len - 1:
                self._obs_array[t] = self.v_go
            # else: blank (zeros)

        self._gt_array = np.zeros(self.trial_len, dtype=np.int64)
        self._gt_array[self.trial_len - 1] = gt
        self.trial = {"ground_truth": gt}
        return gt

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        gt = self._new_trial()
        obs = self._obs_array[self._t_in_trial].copy()
        return obs, {"gt": gt, "new_trial": True}

    def step(self, action):
        is_response = self._t_in_trial == self.trial_len - 1

        if is_response:
            gt = self.trial["ground_truth"]
            reward = 1.0 if int(action) == gt else -1.0
            # Episode terminates — one trial per episode
            obs = np.zeros(self.obs_dim, dtype=np.float32)
            info = {"gt": gt, "new_trial": True}
            return obs, reward, True, False, info

        reward = 0.0
        self._t_in_trial += 1
        obs = self._obs_array[self._t_in_trial].copy()
        info = {"gt": self.trial["ground_truth"], "new_trial": False}
        return obs, reward, False, False, info


gym.register(
    id="TemporalOrderRand-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
)

gym.register(
    id="TemporalOrderRand10-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
    kwargs={"n_steps": 10},
)

gym.register(
    id="TemporalOrderRand20-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
    kwargs={"n_steps": 20},
)

# Small obs_dim=8 variants
gym.register(
    id="TemporalOrderRandS-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
    kwargs={"obs_dim": 8},
)

gym.register(
    id="TemporalOrderRandS10-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
    kwargs={"n_steps": 10, "obs_dim": 8},
)

gym.register(
    id="TemporalOrderRandS20-v0",
    entry_point="temporal_order_env:TemporalOrderRandEnv",
    kwargs={"n_steps": 20, "obs_dim": 8},
)


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------

if __name__ == "__main__":
    env = TemporalOrderEnv(n_steps=5)

    print("=== TemporalOrderEnv smoke test ===")
    print(f"Trial length: {env.trial_len}  (n_steps={env.n_steps})")
    print(f"Obs space:    {env.observation_space}")
    print(f"Action space: {env.action_space}\n")

    # Oracle and random accuracy over many trials
    n_trials = 2000
    oracle_correct = 0
    random_correct = 0

    obs, info = env.reset()
    trials_seen = 0
    while trials_seen < n_trials:
        gt = env.trial["ground_truth"]
        obs, reward, _, _, info = env.step(gt)  # oracle
        if info["new_trial"]:
            if reward == 1.0:
                oracle_correct += 1
            trials_seen += 1

    obs, info = env.reset()
    trials_seen = 0
    while trials_seen < n_trials:
        obs, reward, _, _, info = env.step(env.action_space.sample())
        if info["new_trial"]:
            if reward == 1.0:
                random_correct += 1
            trials_seen += 1

    print(
        f"Oracle accuracy: {oracle_correct}/{n_trials} = {oracle_correct/n_trials:.1%}  (expect 100%)"
    )
    print(
        f"Random accuracy: {random_correct}/{n_trials} = {random_correct/n_trials:.1%}  (expect ~50%)"
    )

    print("\n=== Seeding reproducibility ===")

    def collect_trial_sequence(seed, n=5):
        e = TemporalOrderEnv(n_steps=5)
        e.reset(seed=seed)
        trials = []
        seen = 0
        while seen < n:
            _, _, _, _, info = e.step(0)
            if info["new_trial"]:
                trials.append((e._s1_step, e._s2_step))
                seen += 1
        return trials

    run1 = collect_trial_sequence(seed=42)
    run2 = collect_trial_sequence(seed=42)
    run3 = collect_trial_sequence(seed=99)
    print(f"  seed=42 run1: {run1}")
    print(f"  seed=42 run2: {run2}  (should match run1)")
    print(f"  seed=99 run3: {run3}  (should differ)")
    print(f"  run1 == run2: {run1 == run2}  |  run1 == run3: {run1 == run3}")

    print("\n=== Two sample trials ===")
    obs, info = env.reset()
    for trial_num in range(2):
        print(
            f"\n  Trial {trial_num+1}: S1@step{env._s1_step}, S2@step{env._s2_step}, GT={env.trial['ground_truth']}"
        )
        print(f"    t=0  obs={obs}")
        t = 1
        while True:
            obs, reward, _, _, info = env.step(env.trial["ground_truth"])
            r_str = f"{reward:+.0f}" if reward != 0.0 else " 0"
            cue = " <-- response" if obs[2] == 1.0 else ""
            print(f"    t={t}  obs={obs}  reward={r_str}{cue}")
            t += 1
            if info["new_trial"]:
                break
