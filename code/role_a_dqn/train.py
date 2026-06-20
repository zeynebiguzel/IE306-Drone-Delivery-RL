import gymnasium as gym
import drone_dispatch_env
import numpy as np

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

state_size = 581
action_size = 169

model = DQNNetwork(state_size, action_size)

dummy_state = torch.randn(1, state_size)

q_values = model(dummy_state)

print("\nQ VALUES SHAPE:")
print(q_values.shape)