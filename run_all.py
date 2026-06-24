import gymnasium as gym
import drone_dispatch_env
from drone_dispatch_env import Config, evaluate, RandomPolicy, GreedyNearest, MILPRolling
import numpy as np
import sys
import os

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "code"
    )
)

from role_a_dqn.policies import (
    TrainedVanillaDQNPolicy,
    TrainedDoubleDQNPolicy,
    TrainedDuelingDQNPolicy
)

def main():
    eval_seeds = [0, 1, 2]
    config = Config()
    
    env = gym.make("DroneDispatch-v0")
    obs, info = env.reset(seed=0)
    
    drones_flat = obs["drones"].flatten()
    orders_flat = obs["orders"].flatten()
    grid_flat = obs["grid"].flatten()
    time_flat = obs["time"].flatten()
    state_size = len(drones_flat) + len(orders_flat) + len(grid_flat) + len(time_flat)
    action_size = len(obs["action_mask"])

    print("\n" + "="*60)
   
    print("      IE 306 - DRONE DELIVERY EXPERIMENTATION DASHBOARD")
    print("="*60)
    print(f"Evaluating policies across seeds: {eval_seeds}\n")

    policies = {
        "Random Baseline": RandomPolicy(config),
        "Greedy Nearest (The Bar)": GreedyNearest(config),
        "MILP Rolling (Strong Classical)": MILPRolling(config),
        "Trained Vanilla DQN": TrainedVanillaDQNPolicy(state_size, action_size, "weights/dqn_seed0.pt"),
        "Trained Double DQN": TrainedDoubleDQNPolicy(state_size, action_size, "weights/double_dqn_seed0.pt"),
        "Trained Dueling DQN": TrainedDuelingDQNPolicy(state_size, action_size, "weights/dueling_dqn_seed0.pt")
    }

    print(f"{'Policy Name':<35} | {'Mean Cost/Order':<15} | {'Success Rate':<12} | {'Depletions':<10}")
    print("-" * 80)

    for name, policy in policies.items():
        try:
            results = evaluate(policy, config, seeds=eval_seeds)
            mean_metrics = results["mean"]
            
            cost_per_order = mean_metrics.get("cost_per_order", 0.0)
            success_rate = mean_metrics.get("success_rate", 0.0)
            depletions = mean_metrics.get("depletion_events", 0)
            
            success_str = f"{success_rate * 100:.1f}%" if success_rate <= 1.0 else f"{success_rate:.1f}%"
            
            print(f"{name:<35} | {cost_per_order:<15.2f} | {success_str:<12} | {int(depletions):<10}")
            
        except Exception as e:
            print(f"{name:<35} | [ERROR] Weights not found or incompatible: {str(e)}")

    print("=" * 80)
    print("[INFO] Evaluation complete. Primary objective: Beat Greedy Nearest on Mean Cost/Order.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
