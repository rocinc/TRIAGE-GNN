import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set all random sources to the given seed for reproducibility.

    Note: full cross-version / cross-platform reproducibility is not
    guaranteed even with all seeds fixed. Always log torch.__version__
    and CUDA version alongside results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """Per-worker seed initializer for DataLoader reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
