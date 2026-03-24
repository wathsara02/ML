from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from omi_env import encoding, rules


def encode_central_state(state: dict) -> torch.Tensor:
    """
    Encode centralized state for the critic.

    Args:
        state: dict from env.state()
    """
    hands = state["hands"]
    trump = state.get("trump_suit")
    lead = state.get("lead_suit")
    current_trick = state.get("current_trick", [])
    history = state.get("history", [])
    tricks_won = state.get("tricks_won", (0, 0))

    # Reuse encoding helpers instead of duplicating one-hot logic here
    hand_vecs = []
    for h in hands:
        vec = encoding.card_one_hot(h[0]) * 0.0  # zero vector of right size
        vec = [0.0] * rules.NUM_CARDS
        for c in h:
            vec[c] = 1.0
        hand_vecs.append(vec)

    trump_vec = (
        encoding.one_hot(rules.SUITS.index(trump), 4).tolist()
        if trump is not None
        else [0.0] * 4
    )
    lead_vec = (
        encoding.one_hot(rules.SUITS.index(lead), 4).tolist()
        if lead is not None
        else [0.0] * 4
    )

    trick_vecs = []
    for _, card_idx in current_trick:
        trick_vecs.append(encoding.card_one_hot(card_idx).tolist())
    while len(trick_vecs) < 4:
        trick_vecs.append([0.0] * rules.NUM_CARDS)
    trick_flat = [x for vec in trick_vecs for x in vec]

    score_vec = [
        tricks_won[0] / float(rules.TRICKS_PER_HAND),
        tricks_won[1] / float(rules.TRICKS_PER_HAND),
    ]

    hist_arr = encoding.encode_history(history).reshape(-1)
    features = (
        hand_vecs[0] + hand_vecs[1] + hand_vecs[2] + hand_vecs[3]
        + trump_vec + lead_vec + trick_flat + score_vec
    )
    return torch.tensor(features + hist_arr.tolist(), dtype=torch.float32)


class CentralCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 128):
        super().__init__()
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x).squeeze(-1)
