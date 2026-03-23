from __future__ import annotations

"""Policy network.

Important note:
This project intentionally defaults to a **feed-forward** policy (recurrent_type="none").
The environment already provides explicit history features (played cards / trick context),
so an RNN is not required for strong performance and it greatly simplifies PPO training.

If you want to re-enable recurrence later, implement *sequence-based* PPO updates.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PolicyNet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        history_dim: int,
        action_dim: int,
        hidden_size: int = 128,
        recurrent_type: str = "none",
    ):
        super().__init__()
        self.recurrent_type = recurrent_type.lower()

        # Encoders
        self.obs_encoder = nn.Linear(obs_dim, hidden_size)
        self.hist_encoder = nn.Linear(history_dim, hidden_size)

        # Feed-forward "core" (no RNN by default)
        self.core = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden_size, action_dim)

    # --- RNN compatibility stubs ---
    def init_hidden(self, batch_size: int, device: torch.device):
        """Kept for backward compatibility with inference/training code.

        For recurrent_type="none", there is no hidden state.
        """
        return None

    def forward(
        self,
        obs: torch.Tensor,
        history: torch.Tensor,
        hidden_state=None,
        action_mask: Optional[torch.Tensor] = None,
    ):
        """Forward pass.

        Args:
            obs: (B, obs_dim)
            history: (B, H, F) or flattened (B, history_dim)
            hidden_state: ignored for recurrent_type="none" (returned as None)
            action_mask: (B, action_dim) with 1 for legal, 0 for illegal
        """
        if history.dim() == 3:
            history = history.reshape(history.shape[0], -1)

        obs_emb = torch.tanh(self.obs_encoder(obs))
        hist_emb = torch.tanh(self.hist_encoder(history))
        x = torch.cat([obs_emb, hist_emb], dim=-1)

        x = self.core(x)
        logits = self.actor(x)
        if action_mask is not None:
            logits = mask_logits(logits, action_mask)
        return logits, None


def mask_logits(logits: torch.Tensor, mask: torch.Tensor, mask_value: float = -1e9) -> torch.Tensor:
    """Apply an action mask to logits.

    mask is expected to be 1 for legal actions, 0 for illegal actions.
    """
    expanded_mask = (mask > 0).float()
    return logits + (1.0 - expanded_mask) * mask_value
