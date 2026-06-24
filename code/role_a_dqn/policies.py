import torch
import numpy as np
import gymnasium as gym

from role_a_dqn.network import DQNNetwork
from role_a_dqn.dueling_network import DuelingDQNNetwork

def preprocess_state(obs):
    drones = obs["drones"].flatten()
    orders = obs["orders"].flatten()
    grid = obs["grid"].flatten()
    time = obs["time"].flatten()
    return np.concatenate([drones, orders, grid, time])

# DQN POLICY INTERFACE
class TrainedVanillaDQNPolicy:
    def __init__(self, state_size, action_size, weights_path):
        self.model = DQNNetwork(state_size, action_size)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, obs):
        state = preprocess_state(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = obs["action_mask"]
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        # Ceza almayı engeller
        masked_q_values = q_values.clone()
        masked_q_values[0][action_mask == 0] = -1e9
        return torch.argmax(masked_q_values).item()


# 2. DOUBLE DQN POLICY INTERFACE
class TrainedDoubleDQNPolicy:
    def __init__(self, state_size, action_size, weights_path):
        self.model = DQNNetwork(state_size, action_size)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, obs):
        state = preprocess_state(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = obs["action_mask"]
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        masked_q_values = q_values.clone()
        masked_q_values[0][action_mask == 0] = -1e9
        return torch.argmax(masked_q_values).item()


# 3. DUELING DQN POLICY INTERFACE
class TrainedDuelingDQNPolicy:
    def __init__(self, state_size, action_size, weights_path):
        self.model = DuelingDQNNetwork(state_size, action_size)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def act(self, obs):
        state = preprocess_state(obs)
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = obs["action_mask"]
        
        with torch.no_grad():
            q_values = self.model(state_tensor)
        
        masked_q_values = q_values.clone()
        masked_q_values[0][action_mask == 0] = -1e9
        return torch.argmax(masked_q_values).item()
