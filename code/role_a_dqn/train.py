import gymnasium as gym
import drone_dispatch_env
import numpy as np
import torch

from network import DQNNetwork


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

# State ve action boyutlarını belirle
state = preprocess_state(obs)

state_size = len(state)
action_size = len(obs["action_mask"])

print("STATE SIZE:", state_size)
print("ACTION SIZE:", action_size)

# DQN ağı
model = DQNNetwork(state_size, action_size)

# Episode değişkenleri
terminated = False
truncated = False

total_reward = 0
step_count = 0

random_actions = 0
greedy_actions = 0

# Episode döngüsü
while not terminated and not truncated:

    state = preprocess_state(obs)

    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    q_values = model(state_tensor)

    epsilon = 0.1

    if np.random.random() < epsilon:

       valid_actions = np.where(obs["action_mask"] == 1)[0]
       action = np.random.choice(valid_actions)
       random_actions += 1

    else:

       action = torch.argmax(q_values).item()
       greedy_actions += 1

    next_obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    obs = next_obs

print("\nEPISODE FINISHED")
print("TOTAL REWARD:", total_reward)
print("STEPS:", step_count)

print("RANDOM ACTIONS:", random_actions)
print("GREEDY ACTIONS:", greedy_actions)