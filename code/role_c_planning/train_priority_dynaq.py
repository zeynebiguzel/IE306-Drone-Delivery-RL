import gymnasium as gym
import drone_dispatch_env

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

import argparse
import random
import yaml

from network import DynaQNetwork
from priority_dynaq import PriorityReplayBuffer


def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def preprocess_state(obs):
    drones = obs["drones"].flatten()
    orders = obs["orders"].flatten()
    grid = obs["grid"].flatten()
    time = obs["time"].flatten()

    return np.concatenate(
        [drones, orders, grid, time]
    )


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
seed = args.seed

set_seed(seed)

with open("configs/priority_dynaq.yaml", "r") as f:
    config = yaml.safe_load(f)

num_episodes = config["num_episodes"]
batch_size = config["batch_size"]
epsilon = config["epsilon"]
gamma = config["gamma"]
learning_rate = config["learning_rate"]
target_update_frequency = config["target_update_frequency"]

# Role C contribution
planning_steps = config["planning_steps"]

env = gym.make("DroneDispatch-v0")

obs, info = env.reset(seed=seed)

state_size = len(preprocess_state(obs))
action_size = len(obs["action_mask"])

print("STATE SIZE:", state_size)
print("ACTION SIZE:", action_size)

# Online Network
model = DynaQNetwork(
    state_size,
    action_size
)

# Target Network
target_model = DynaQNetwork(
    state_size,
    action_size
)

target_model.load_state_dict(
    model.state_dict()
)

target_model.eval()

optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)

buffer = PriorityReplayBuffer()

episode_rewards = []

for episode in range(num_episodes):

    obs, info = env.reset(
        seed=seed + episode
    )

    terminated = False
    truncated = False

    total_reward = 0

    while not terminated and not truncated:

        state = preprocess_state(obs)

        state_tensor = torch.FloatTensor(
            state
        ).unsqueeze(0)

        q_values = model(state_tensor)

        # Epsilon Greedy
        if np.random.random() < epsilon:

            valid_actions = np.where(
                obs["action_mask"] == 1
            )[0]

            action = np.random.choice(
                valid_actions
            )

        else:

            action_mask = torch.FloatTensor(
                obs["action_mask"]
            )

            masked_q_values = q_values.clone()

            masked_q_values[0][
                action_mask == 0
            ] = -1e9

            action = torch.argmax(
                masked_q_values
            ).item()

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        next_state = preprocess_state(next_obs)

        # TD ERROR
        with torch.no_grad():

            current_q = q_values[
                0,
                action
            ].item()

            next_state_tensor = torch.FloatTensor(
                next_state
            ).unsqueeze(0)

            next_q_values = target_model(
                next_state_tensor
            )

            next_q = torch.max(
                next_q_values
            ).item()

            target = reward + gamma * next_q * (1 - done)

            td_error = abs(
                target - current_q
            )
        # URGENCY
        ages = obs["orders"][:, 4]

        if len(ages) > 0:
            urgency = np.max(ages) / 60.0
        else:
            urgency = 0.0
        # PRIORITY SCORE
        priority = td_error * (1.0 + urgency)
        
        buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            priority
        )

        total_reward += reward

        obs = next_obs

        # REAL EXPERIENCE UPDATE
        if len(buffer.memory) >= batch_size:

            batch = buffer.sample(
                batch_size
            )

            states = torch.FloatTensor(
                np.array([x[0] for x in batch])
            )

            actions = torch.LongTensor(
                np.array([x[1] for x in batch])
            )

            rewards = torch.FloatTensor(
                np.array([x[2] for x in batch])
            )

            next_states = torch.FloatTensor(
                np.array([x[3] for x in batch])
            )

            dones = torch.FloatTensor(
                np.array([x[4] for x in batch])
            )

            q_values_batch = model(states)

            current_q = q_values_batch.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()

            with torch.no_grad():

                next_q_values = target_model(
                    next_states
                )

                max_next_q = next_q_values.max(
                    dim=1
                )[0]

                target_q = rewards + gamma * max_next_q * (
                    1 - dones
                )

            loss = F.mse_loss(
                current_q,
                target_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # DYNA-Q PLANNING
            for _ in range(planning_steps):

                planning_batch = buffer.sample(
                    batch_size
                )

                if len(planning_batch) == 0:
                    continue

                states = torch.FloatTensor(
                    np.array([x[0] for x in planning_batch])
                )

                actions = torch.LongTensor(
                    np.array([x[1] for x in planning_batch])
                )

                rewards = torch.FloatTensor(
                    np.array([x[2] for x in planning_batch])
                )

                next_states = torch.FloatTensor(
                    np.array([x[3] for x in planning_batch])
                )

                dones = torch.FloatTensor(
                    np.array([x[4] for x in planning_batch])
                )

                q_values_batch = model(states)

                current_q = q_values_batch.gather(
                    1,
                    actions.unsqueeze(1)
                ).squeeze()

                with torch.no_grad():

                    next_q_values = target_model(
                        next_states
                    )

                    max_next_q = next_q_values.max(
                        dim=1
                    )[0]

                    target_q = rewards + gamma * max_next_q * (
                        1 - dones
                    )

                planning_loss = F.mse_loss(
                    current_q,
                    target_q
                )

                optimizer.zero_grad()
                planning_loss.backward()
                optimizer.step()

    if (episode + 1) % target_update_frequency == 0:

        target_model.load_state_dict(
            model.state_dict()
        )

    episode_rewards.append(
        total_reward
    )

    print(
        f"Episode {episode + 1} | "
        f"Reward = {total_reward:.2f} | "
        f"Buffer = {len(buffer.memory)}"
    )

torch.save(
    model.state_dict(),
    f"weights/priority_dynaq_seed{seed}.pt"
)

with open(
    f"logs/priority_dynaq_seed{seed}.csv",
    "w"
) as f:

    f.write("episode,reward\n")

    for i, reward in enumerate(
        episode_rewards,
        start=1
    ):
        f.write(
            f"{i},{reward}\n"
        )

print(
    f"MODEL SAVED: weights/priority_dynaq_seed{seed}.pt"
)

print(
    "\nPRIORITY DYNA-Q TRAINING FINISHED"
)