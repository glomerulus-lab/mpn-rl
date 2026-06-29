"""Global RNG seeding for reproducible runs."""

import random

import numpy as np
import torch


def seed_rngs(seed: int) -> None:
    """Seed Python, NumPy, and Torch global RNGs for a reproducible run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
