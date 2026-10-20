"""
flavri_mcp.config
Configuration settings, default paths, and device management.
"""

import os
from pathlib import Path
import torch

# Base directories
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

# Data directory
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Cache path
CACHE_PATH = Path(os.environ.get("FLAVRI_CACHE_PATH", DATA_DIR / "flavri_cache.pt"))

# Checkpoint candidate locations
DEFAULT_CHECKPOINT_CANDIDATES = [
    Path(os.environ.get("FLAVRI_CHECKPOINT", "")),
    PROJECT_ROOT / "checkpoints" / "flavri-0.8.pt",
    PROJECT_ROOT / "data" / "flavri-0.8.pt",
    Path.home() / "Downloads" / "flavri-backup" / "flavri-0.8.pt",
    PROJECT_ROOT.parent / "flavri" / "checkpoints" / "flavri-0.8.pt",
]

def get_checkpoint_path() -> Path:
    """Finds the best available FLAVRI 0.8 model checkpoint."""
    for cand in DEFAULT_CHECKPOINT_CANDIDATES:
        if str(cand) and cand.exists() and cand.is_file():
            return cand
    raise FileNotFoundError(
        "FLAVRI checkpoint (flavri-0.8.pt) not found. "
        "Please place flavri-0.8.pt in Downloads/flavri-backup/ or set FLAVRI_CHECKPOINT."
    )

def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    device_str = os.environ.get("FLAVRI_DEVICE", "")
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
