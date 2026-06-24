import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from network import DQNNetwork
from replay_buffer import ReplayBuffer


def preprocess_state(obs):

    drones = obs["drones"].flatten()
    orders = obs["orders"].flatten()
    grid = obs["grid"].flatten()
    time = obs["time"].flatten()

    state = np.concatenate([
        drones,
        orders,
        grid,
        time
    ])

    return state


# Environment oluştur
env = gym.make("DroneDispatch-v0")

# Reset
obs, info = env.reset(seed=0)

# State ve action boyutları
state = preprocess_state(obs)

state_size = len(state)
action_size = len(obs["action_mask"])

print("STATE SIZE:", state_size)
print("ACTION SIZE:", action_size)

# DQN ağı
model = DQNNetwork(state_size, action_size)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Replay Buffer
buffer = ReplayBuffer()

# Episode değişkenleri
terminated = False
truncated = False

total_reward = 0
step_count = 0

random_actions = 0
greedy_actions = 0

epsilon = 0.1

# Episode döngüsü
while not terminated and not truncated:

    state = preprocess_state(obs)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    q_values = model(state_tensor)

    # Epsilon-Greedy
    if np.random.random() < epsilon:

        valid_actions = np.where(obs["action_mask"] == 1)[0]
        action = np.random.choice(valid_actions)

        random_actions += 1

    else:

        action = torch.argmax(q_values).item()

        greedy_actions += 1

    # Environment adımı
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
    step_count += 1

    obs = next_obs

# Buffer'dan örnek çek
batch_size = 32

sample_batch = buffer.sample(batch_size)

print("\nBATCH SIZE:")
print(len(sample_batch))

first_exp = sample_batch[0]

print("\nFIRST EXPERIENCE:")
print("STATE SHAPE:", first_exp[0].shape)
print("ACTION:", first_exp[1])
print("REWARD:", first_exp[2])
print("NEXT STATE SHAPE:", first_exp[3].shape)
print("DONE:", first_exp[4])

# Tensorlara çevir
states = np.array([exp[0] for exp in sample_batch])
actions = np.array([exp[1] for exp in sample_batch])
rewards = np.array([exp[2] for exp in sample_batch])

states = torch.FloatTensor(states)
actions = torch.LongTensor(actions)
rewards = torch.FloatTensor(rewards)

# Q değerleri
q_values = model(states)

chosen_q_values = q_values.gather(
    1,
    actions.unsqueeze(1)
).squeeze()

loss = F.mse_loss(
    chosen_q_values,
    rewards
)

optimizer.zero_grad()

loss.backward()

optimizer.step()

print("\nLOSS:")
print(loss.item())

# Sonuçlar
print("\nEPISODE FINISHED")
print("TOTAL REWARD:", total_reward)
print("STEPS:", step_count)

print("RANDOM ACTIONS:", random_actions)
print("GREEDY ACTIONS:", greedy_actions)

print("BUFFER SIZE:", len(buffer))