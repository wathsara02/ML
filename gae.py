from __future__ import annotations

from typing import List, Tuple

import torch


def compute_gae(rewards: List[float], values: List[float], dones: List[bool], gamma: float, lam: float):
    """
    Generic GAE calculation for a single trajectory.
    Returns advantages and returns tensors.
    """
    advs = []
    gae = 0.0
    next_value = 0.0
    for step in reversed(range(len(rewards))):
        mask = 0.0 if dones[step] else 1.0
        delta = rewards[step] + gamma * next_value * mask - values[step]
        gae = delta + gamma * lam * mask * gae
        advs.insert(0, gae)
        next_value = values[step]
    adv_tensor = torch.tensor(advs, dtype=torch.float32)
    ret_tensor = adv_tensor + torch.tensor(values, dtype=torch.float32)
    return adv_tensor, ret_tensor
