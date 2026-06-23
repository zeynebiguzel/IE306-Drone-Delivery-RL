import torch
import torch.nn as nn


class DuelingDQNNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):

        features = self.feature(x)

        value = self.value_stream(features)

        advantage = self.advantage_stream(features)

        q_values = (
            value
            + advantage
            - advantage.mean(
                dim=1,
                keepdim=True
            )
        )

        return q_values