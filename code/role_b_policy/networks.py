"""
Neural network models for Role B.

Role B uses policy-based and actor-critic methods:
- REINFORCE
- A2C with GAE / advantage normalization
- DDPG for the continuous control sub-environment

This file defines the Actor-Critic network used for DroneDispatch-v0.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def flatten_dispatch_obs(obs: Dict[str, Any]) -> np.ndarray:
    """
    Convert the DroneDispatch-v0 dictionary observation into a flat vector.

    The environment observation contains:
    - drones: drone positions, battery levels, status information
    - orders: visible pending orders
    - time: normalized simulation time

    We do not include the grid here in the first simple version.
    The grid is static and can be added later if needed.
    """
    drones = np.asarray(obs["drones"], dtype=np.float32).flatten()
    orders = np.asarray(obs["orders"], dtype=np.float32).flatten()
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    return np.concatenate([drones, orders, time]).astype(np.float32)


class ActorCritic(nn.Module):
    """
    Shared-body Actor-Critic network for DroneDispatch-v0.

    Input:
        Flattened observation vector.

    Outputs:
        actor_logits: one logit per discrete action
        value: scalar state-value estimate V(s)
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_size: int = 128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.actor = nn.Linear(hidden_size, n_actions)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, obs_tensor: torch.Tensor):
        x = self.shared(obs_tensor)
        actor_logits = self.actor(x)
        value = self.critic(x).squeeze(-1)
        return actor_logits, value

    def get_action_and_value(self, obs_tensor: torch.Tensor, action_mask: torch.Tensor):
        """
        Select an action using masked categorical sampling.

        Invalid actions must not be selected. Therefore, logits of invalid
        actions are replaced with a very negative number before the distribution
        is created.
        """
        logits, value = self.forward(obs_tensor)

        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        dist = Categorical(logits=masked_logits)

        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def get_value(self, obs_tensor: torch.Tensor):
        """
        Return only the critic value V(s).
        """
        _, value = self.forward(obs_tensor)
        return value