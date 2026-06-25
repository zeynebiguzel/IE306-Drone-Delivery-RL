import torch
import numpy as np

from network import DynaQNetwork


def preprocess_state(obs):
    drones = obs["drones"].flatten()
    orders = obs["orders"].flatten()
    grid = obs["grid"].flatten()
    time = obs["time"].flatten()

    return np.concatenate(
        [drones, orders, grid, time]
    )


class TrainedPriorityDynaQPolicy:

    def __init__(
        self,
        state_size,
        action_size,
        weights_path
    ):

        self.model = DynaQNetwork(
            state_size,
            action_size
        )

        self.model.load_state_dict(
            torch.load(
                weights_path,
                map_location=torch.device("cpu")
            )
        )

        self.model.eval()

    def act(self, obs):

        state = preprocess_state(obs)

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        action_mask = obs["action_mask"]

        with torch.no_grad():

            q_values = self.model(
                state_tensor
            )

        masked_q_values = q_values.clone()

        masked_q_values[
            0
        ][action_mask == 0] = -1e9

        return torch.argmax(
            masked_q_values
        ).item()