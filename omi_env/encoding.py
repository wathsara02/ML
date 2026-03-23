"""
Observation and action encoding utilities for the Omi environment.

The observation structure mirrors PettingZoo's AEC/AIO pattern:
{
    "observation": np.ndarray,  # flat vector for policy input
    "action_mask": np.ndarray,  # legal moves (includes trump declaration)
    "history": np.ndarray       # sequence features for recurrent policies
}
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import rules

# History configuration: number of past plays to encode
HISTORY_LEN = 32  # total plays in a 32-card hand (8 tricks × 4 plays)
HISTORY_FEAT_DIM = rules.NUM_CARDS + 12  # card one-hot + 4 player + 4 lead + 4 trump


def card_one_hot(card_idx: int) -> np.ndarray:
    vec = np.zeros(rules.NUM_CARDS, dtype=np.float32)
    vec[card_idx] = 1.0
    return vec


def one_hot(idx: int, size: int) -> np.ndarray:
    vec = np.zeros(size, dtype=np.float32)
    vec[idx] = 1.0
    return vec


def encode_history(
    history: Sequence[Tuple[int, int, Optional[str], Optional[str]]]
) -> np.ndarray:
    """
    Encode a list of past plays into a fixed-length sequence.

    Args:
        history: iterable of tuples (player_id, card_idx, lead_suit, trump_suit)
            ordered from oldest to newest.
    """
    encoded = np.zeros((HISTORY_LEN, HISTORY_FEAT_DIM), dtype=np.float32)
    start = max(0, len(history) - HISTORY_LEN)
    slice_hist = history[start:]
    offset = HISTORY_LEN - len(slice_hist)
    for i, (player, card_idx, lead_suit, trump_suit) in enumerate(slice_hist):
        row = np.concatenate(
            [
                card_one_hot(card_idx),
                one_hot(player, 4),
                one_hot(rules.SUITS.index(lead_suit), 4)
                if lead_suit is not None
                else np.zeros(4, dtype=np.float32),
                one_hot(rules.SUITS.index(trump_suit), 4)
                if trump_suit is not None
                else np.zeros(4, dtype=np.float32),
            ]
        )
        encoded[offset + i] = row
    return encoded


def encode_observation(
    agent_id: int,
    hand: Sequence[int],
    trump_suit: Optional[str],
    lead_suit: Optional[str],
    current_trick: Sequence[Tuple[int, int]],
    scores: Tuple[int, int],
    action_mask: Sequence[int],
    history: Sequence[Tuple[int, int, Optional[str], Optional[str]]],
) -> dict:
    """
    Build the observation dictionary for the current agent.

    Args:
        agent_id: active agent id (0-3).
        hand: list of card indices for the agent.
        trump_suit: current trump suit or None if not declared yet.
        lead_suit: suit of the current trick leader.
        current_trick: list of (player_id, card_idx) tuples already played in this trick.
        scores: tuple of team scores (team 0, team 1).
        action_mask: legal action mask aligned with rules.ACTION_DIM.
        history: past play tuples (player_id, card_idx, lead_suit, trump_suit).
    """
    hand_vec = np.zeros(rules.NUM_CARDS, dtype=np.float32)
    for c in hand:
        hand_vec[c] = 1.0

    trump_vec = (
        one_hot(rules.SUITS.index(trump_suit), 4) if trump_suit is not None else np.zeros(4, dtype=np.float32)
    )
    lead_vec = (
        one_hot(rules.SUITS.index(lead_suit), 4) if lead_suit is not None else np.zeros(4, dtype=np.float32)
    )

    trick_vecs: List[np.ndarray] = []
    # Up to 4 cards per trick; pad to 4
    for _, card_idx in current_trick:
        trick_vecs.append(card_one_hot(card_idx))
    while len(trick_vecs) < 4:
        trick_vecs.append(np.zeros(rules.NUM_CARDS, dtype=np.float32))
    trick_flat = np.concatenate(trick_vecs, axis=0)

    # normalize to [0,1] for an 8-trick hand
    score_vec = np.array(scores, dtype=np.float32) / float(rules.TRICKS_PER_HAND)
    player_vec = one_hot(agent_id, 4)

    observation_vec = np.concatenate(
        [
            hand_vec,
            trump_vec,
            lead_vec,
            trick_flat,
            score_vec,
            player_vec,
        ],
        axis=0,
    ).astype(np.float32)

    return {
        "observation": observation_vec,
        "action_mask": np.array(action_mask, dtype=np.float32),
        "history": encode_history(history),
    }


def decode_action(action: int) -> Tuple[bool, int]:
    """
    Decode an action index.

    Returns:
        (is_trump_action, payload) where payload is suit index (0-3) if trump,
        otherwise card index (0-{rules.NUM_CARDS - 1}).
    """
    if rules.is_trump_action(action):
        return True, action - rules.ACTION_TRUMP_OFFSET
    if action < 0 or action >= rules.NUM_CARDS:
        raise ValueError(f"Invalid action {action}")
    return False, action


def observation_length() -> int:
    """Length of the flat observation vector."""
    # hand one-hot + trump 4 + lead 4 + current trick (4 cards) + score (2) + player id (4)
    return rules.NUM_CARDS + 4 + 4 + (4 * rules.NUM_CARDS) + 2 + 4
