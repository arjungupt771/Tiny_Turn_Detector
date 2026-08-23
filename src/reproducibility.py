"""Small reproducibility helpers shared by experiments."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_config(path: str | Path) -> dict:
    import yaml
    with Path(path).open(encoding="utf-8") as handle: return yaml.safe_load(handle)
