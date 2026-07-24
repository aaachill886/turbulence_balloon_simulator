from __future__ import annotations

import torch
import torch.nn as nn


class BalloonPolicyNet(nn.Module):
    """Stage 3 policy network.

    Input:  state_vector (B, D)
    Output: action (B, 3) for direct [x, y, z] control deltas
    """

    def __init__(self, state_dim: int = 14, hidden_dim: int = 256, action_dim: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        x = self.backbone(state)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value
