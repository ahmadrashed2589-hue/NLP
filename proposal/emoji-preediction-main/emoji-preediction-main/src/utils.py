import os
import random
import numpy as np
import torch


def create_folders():
    folders = [
        "data",
        "models",
        "models/bert_model",
        "outputs",
        "outputs/figures",
        "outputs/reports",
        "app",
        "src"
    ]

    for folder in folders:
        os.makedirs(folder, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)