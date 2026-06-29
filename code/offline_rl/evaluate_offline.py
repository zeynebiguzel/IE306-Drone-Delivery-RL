from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from drone_dispatch_env import Config
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling


# ------------------------------------------------------------
# Observation processing
# ------------------------------------------------------------

def flatten_full_obs(obs: Dict[str, Any]) -> np.ndarray:
    drones = np.asarray(obs["drones"], dtype=np.float32).flatten()
    orders = np.asarray(obs["orders"], dtype=np.float32).flatten()
    grid = np.asarray(obs["grid"], dtype=np.float32).flatten()
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    return np.concatenate([drones, orders, grid, time]).astype(np.float32)


# ------------------------------------------------------------
# Network
# ------------------------------------------------------------

class OfflineQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# Offline policy wrapper
# ------------------------------------------------------------

class OfflinePolicy:
    def __init__(self, weights_path: Path):
        self.device = torch.device("cpu")

        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)

        self.model_type = checkpoint.get("model_type", "offline_model")
        self.state_dim = int(checkpoint["state_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.hidden_size = int(checkpoint.get("hidden_size", 256))

        self.model = OfflineQNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, obs):
        state = flatten_full_obs(obs)
        action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            values = self.model(x).cpu().numpy()[0]

        values[action_mask <= 0] = -1e9

        return int(np.argmax(values))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_seeds(text: str):
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def make_baseline(cls, cfg):
    for args in [(cfg,), tuple()]:
        try:
            return cls(*args)
        except Exception:
            pass

    raise RuntimeError(f"Could not initialize baseline: {cls.__name__}")


def print_table(results):
    print("\nOFFLINE RL EVALUATION COMPARISON")
    print("-" * 112)
    print(
        f"{'Policy':<24} | {'Cost/Order':>10} | {'Success':>8} | "
        f"{'On-time':>8} | {'Delivered':>10} | {'Dropped':>8} | "
        f"{'Depletion':>10} | {'Return':>10}"
    )
    print("-" * 112)

    for name, r in results.items():
        print(
            f"{name:<24} | "
            f"{r.get('cost_per_order', 0):>10.4f} | "
            f"{r.get('success_rate', 0):>8.3f} | "
            f"{r.get('ontime_rate', 0):>8.3f} | "
            f"{r.get('n_delivered', 0):>10.2f} | "
            f"{r.get('n_dropped', 0):>8.2f} | "
            f"{r.get('depletion_events', 0):>10.2f} | "
            f"{r.get('episode_return', 0):>10.2f}"
        )

    print("-" * 112)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_standard.yaml",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
    )

    parser.add_argument(
        "--bc-weights",
        type=str,
        default="weights/offline_rl/bc_policy.pt",
    )

    parser.add_argument(
        "--dqn-weights",
        type=str,
        default="weights/offline_rl/offline_dqn.pt",
    )

    parser.add_argument(
        "--cql-weights",
        type=str,
        default="weights/offline_rl/cql_lite.pt",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="logs/offline_rl/offline_eval_results.json",
    )

    parser.add_argument(
        "--with-baselines",
        action="store_true",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.seeds)

    results = {}

    models = [
        ("Behavior Cloning", ROOT / args.bc_weights),
        ("Naive Offline DQN", ROOT / args.dqn_weights),
        ("IQL-lite", ROOT / args.cql_weights),
    ]

    for name, weights_path in models:
        print(f"\nEvaluating {name}")
        print(f"Weights: {weights_path}")

        policy = OfflinePolicy(weights_path)
        metrics = evaluate(policy, cfg, seeds)
        results[name] = metrics

        print(json.dumps(metrics, indent=2))

    if args.with_baselines:
        baseline_specs = [
            ("Random Policy", RandomPolicy),
            ("Greedy Nearest", GreedyNearest),
            ("MILP Rolling", MILPRolling),
        ]

        for name, cls in baseline_specs:
            print(f"\nEvaluating baseline: {name}")

            try:
                policy = make_baseline(cls, cfg)
                metrics = evaluate(policy, cfg, seeds)
                results[name] = metrics
                print(json.dumps(metrics, indent=2))
            except Exception as e:
                print(f"[skip] {name}: {e}")
def get_mean_metrics(result):
    if isinstance(result, dict) and "mean" in result:
        return result["mean"]
    return result


def print_table(results):
    print("\nOFFLINE RL EVALUATION COMPARISON")
    print("-" * 112)
    print(
        f"{'Policy':<24} | {'Cost/Order':>10} | {'Success':>8} | "
        f"{'On-time':>8} | {'Delivered':>10} | {'Dropped':>8} | "
        f"{'Depletion':>10} | {'Return':>10}"
    )
    print("-" * 112)

    for name, r in results.items():
        m = get_mean_metrics(r)

        print(
            f"{name:<24} | "
            f"{m.get('cost_per_order', 0):>10.4f} | "
            f"{m.get('success_rate', 0):>8.3f} | "
            f"{m.get('ontime_rate', 0):>8.3f} | "
            f"{m.get('n_delivered', 0):>10.2f} | "
            f"{m.get('n_dropped', 0):>8.2f} | "
            f"{m.get('depletion_events', 0):>10.2f} | "
            f"{m.get('episode_return', 0):>10.2f}"
        )

    print("-" * 112)

if __name__ == "__main__":
    main()