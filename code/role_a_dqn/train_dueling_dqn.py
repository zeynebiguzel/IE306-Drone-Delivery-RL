import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import random
import yaml

from dueling_network import DuelingDQNNetwork
from replay_buffer import ReplayBuffer

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

    return np.concatenate([
        drones,
        orders,
        grid,
        time
    ])

parser = argparse.ArgumentParser()
parser.add_argument(
    "--seed",
    type=int,
    default=0
)
args = parser.parse_args()
seed = args.seed
set_seed(seed)

with open(
    "configs/dueling_dqn.yaml",
    "r"
) as f:

    config = yaml.safe_load(f)

num_episodes = config["num_episodes"]
batch_size = config["batch_size"]
epsilon = config["epsilon"]
gamma = config["gamma"]
learning_rate = config["learning_rate"]
target_update_frequency = config["target_update_frequency"]

# Environment
env = gym.make("DroneDispatch-v0")
obs, info = env.reset(seed=seed)

state_size = len(preprocess_state(obs))
action_size = len(obs["action_mask"])
print("STATE SIZE:", state_size)
print("ACTION SIZE:", action_size)

# Online Network
model = DuelingDQNNetwork(
    state_size,
    action_size
)
# Target Network
target_model = DuelingDQNNetwork(
    state_size,
    action_size
)
target_model.load_state_dict(
    model.state_dict()
)
target_model.eval()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=learning_rate
)
# Replay Buffer
buffer = ReplayBuffer()
episode_rewards = []

# Training Loop
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

        # Epsilon-Greedy
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

        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(
            state,
            action,
            reward,
            preprocess_state(next_obs),
            done
        )

        total_reward += reward
        obs = next_obs

        # Learning
        if len(buffer) >= batch_size:
            sample_batch = buffer.sample(
                batch_size
            )
            states = torch.FloatTensor(
                np.array(
                    [exp[0] for exp in sample_batch]
                )
            )
            actions = torch.LongTensor(
                np.array(
                    [exp[1] for exp in sample_batch]
                )
            )
            rewards = torch.FloatTensor(
                np.array(
                    [exp[2] for exp in sample_batch]
                )
            )
            next_states = torch.FloatTensor(
                np.array(
                    [exp[3] for exp in sample_batch]
                )
            )
            dones = torch.FloatTensor(
                np.array(
                    [exp[4] for exp in sample_batch]
                )
            )

            # Current Q
            q_values = model(states)
            current_q = q_values.gather(
                1,
                actions.unsqueeze(1)
            ).squeeze()

            # Double DQN Target
            with torch.no_grad():
                next_actions = model(
                    next_states
                ).argmax(dim=1)

                next_q_values = target_model(
                    next_states
                )

                max_next_q = next_q_values.gather(
                    1,
                    next_actions.unsqueeze(1)
                ).squeeze(1)

                target_q = (
                    rewards
                    + gamma * max_next_q * (1 - dones)
                )

            current_q = current_q.view(-1)
            target_q = target_q.view(-1)

            loss = F.mse_loss(
                current_q,
                target_q
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Target Network Update
    if (episode + 1) % target_update_frequency == 0:
        target_model.load_state_dict(
            model.state_dict()
        )

    episode_rewards.append(total_reward)
    print(
        f"Episode {episode + 1} | "
        f"Reward = {total_reward:.2f} | "
        f"Buffer = {len(buffer)}"
    )

torch.save(
    model.state_dict(),
    f"weights/dueling_dqn_seed{seed}.pt"
)
with open(
    f"logs/dueling_dqn_seed{seed}.csv",
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
print("\nTRAINING FINISHED")