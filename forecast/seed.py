"""Make runs repeatable by fixing every source of randomness."""
import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch with the same number.

    Call once at the start of every run (and before each NN training run)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)    # hashing order (affects some set/dict iteration)
    random.seed(seed)                           # Python's own random module
    np.random.seed(seed)                        # NumPy (and anything built on it, e.g. scikit-learn)

    # PyTorch is only needed for the neural networks; skip until it is installed.
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)                     # CPU and all GPUs
    # Force deterministic algorithms where available. 
    torch.use_deterministic_algorithms(True, warn_only=True)