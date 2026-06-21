"""
Gymnasium wrapper to capture and expose NeuroGym info dict.
"""

import gymnasium as gym


class NeuroGymInfoWrapper(gym.Wrapper):
    """
    Wrapper that captures the info dict from each step and stores it
    as an attribute so it can be accessed by TorchRL transforms.

    IMPORTANT: TorchRL's GymEnv inverts the ground truth values (0 <-> 1).
    We invert them back to match the correct NeuroGym semantics.
    """

    def __init__(self, env, invert_gt=False):
        super().__init__(env)
        self._last_info = {}
        self._invert_gt = invert_gt  # Set True when used with TorchRL GymEnv

    def step(self, action):
        # Capture GT BEFORE step (for the trial that's about to complete)
        self._current_trial_gt = self.env.unwrapped.trial.get("ground_truth", None)

        obs, reward, done, truncated, info = self.env.step(action)

        # When new_trial=True, info['gt'] is already for the NEXT trial
        # Replace it with the captured GT from the completed trial
        if info.get("new_trial", False):
            if self._current_trial_gt is not None:
                info["gt"] = self._current_trial_gt

        self._last_info = info
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._current_trial_gt = self.env.unwrapped.trial.get("ground_truth", None)
        self._last_info = info
        return obs, info
