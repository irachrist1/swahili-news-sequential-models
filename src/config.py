"""Shared paths, label names and settings used by every script."""

import random
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_DIR / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
FIGURES_DIR = PROJECT_DIR / "figures"

DATASET_URL = (
    "https://huggingface.co/datasets/community-datasets/swahili_news/"
    "resolve/main/swahili_news/{split}-00000-of-00001.parquet"
)
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.sw.300.vec.gz"

# Label ids follow the order in the original dataset release.
LABEL_NAMES = ["uchumi", "kitaifa", "michezo", "kimataifa", "burudani", "afya"]
LABEL_ENGLISH = {
    "uchumi": "economy",
    "kitaifa": "national",
    "michezo": "sports",
    "kimataifa": "international",
    "burudani": "entertainment",
    "afya": "health",
}
NUM_CLASSES = len(LABEL_NAMES)

SEED = 42
VALIDATION_FRACTION = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_output_dirs():
    for folder in [DATA_DIR, EMBEDDINGS_DIR, RESULTS_DIR, PREDICTIONS_DIR, FIGURES_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
