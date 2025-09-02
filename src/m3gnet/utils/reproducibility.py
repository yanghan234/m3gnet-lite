"""Utilities for ensuring reproducibility."""

import random

import numpy as np
import torch


def ensure_reproducibility(random_seed: int = 42):
    """Ensure reproducibility of random operations."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    # For multi-GPU setups
    torch.cuda.manual_seed_all(random_seed)
    # Add MPS seed
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
