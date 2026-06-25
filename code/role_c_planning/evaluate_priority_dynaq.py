import gymnasium as gym
import drone_dispatch_env

from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env.config import Config

from policy_interface import (
    TrainedPriorityDynaQPolicy
)

env = gym.make(
    "DroneDispatch-v0"
)

obs, info = env.reset(seed=0)

state_size = (
    len(obs["drones"].flatten())
    + len(obs["orders"].flatten())
    + len(obs["grid"].flatten())
    + len(obs["time"].flatten())
)

action_size = len(
    obs["action_mask"]
)

policy = TrainedPriorityDynaQPolicy(
    state_size=state_size,
    action_size=action_size,
    weights_path="weights/priority_dynaq_seed0.pt"
)

cfg = Config.from_yaml(
    "configs/eval_standard.yaml"
)

results = evaluate(
    policy,
    cfg,
    seeds=[0,1,2]
)

print(results["mean"])