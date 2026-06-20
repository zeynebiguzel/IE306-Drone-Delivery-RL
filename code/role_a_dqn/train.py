import gymnasium as gym
import drone_dispatch_env
import numpy as np
#state preprocessing.
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

env = gym.make("DroneDispatch-v0")

obs, info = env.reset(seed=0)

print("OBS KEYS:")
print(obs.keys())

print("\nDRONES SHAPE:")
print(obs["drones"].shape)

print("\nORDERS SHAPE:")
print(obs["orders"].shape)

print("\nGRID SHAPE:")
print(obs["grid"].shape)

print("\nTIME:")
print(obs["time"])

print("\nACTION MASK SHAPE:")
print(obs["action_mask"].shape)

print("\nNUMBER OF VALID ACTIONS:")
print(np.sum(obs["action_mask"]))

print("\nFIRST DRONE:")
print(obs["drones"][0])

print("\nFIRST ORDER:")
print(obs["orders"][0])

print("\nACTION MASK:")
print(obs["action_mask"][:30])

import torch
from network import DQNNetwork

state = preprocess_state(obs)

state_tensor = torch.FloatTensor(state).unsqueeze(0)

state_size = len(state)
action_size = len(obs["action_mask"])

model = DQNNetwork(state_size, action_size)

q_values = model(state_tensor)

print("\nQ VALUES SHAPE:")
print(q_values.shape)

print("\nFIRST 10 Q VALUES:")
print(q_values[0][:10].detach().numpy())

state = preprocess_state(obs)

print("\nSTATE SHAPE:")
print(state.shape)