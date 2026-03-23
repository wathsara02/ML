from __future__ import annotations

import csv
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = False) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv_row(path: str, headers: Tuple[str, ...], row: Dict[str, object]):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def masked_sample(logits: torch.Tensor, mask: torch.Tensor, deterministic: bool = False):
    logits = logits + (1.0 - mask) * -1e9
    probs = torch.softmax(logits, dim=-1)
    if deterministic:
        return torch.argmax(probs, dim=-1), probs
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    return action, probs


def bootstrap_confidence_interval(data, num_bootstrap: int = 2000, alpha: float = 0.05):
    data = np.array(data)
    n = len(data)
    if n == 0:
        return (0.0, 0.0)
    samples = []
    for _ in range(num_bootstrap):
        idx = np.random.randint(0, n, n)
        samples.append(np.mean(data[idx]))
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)
