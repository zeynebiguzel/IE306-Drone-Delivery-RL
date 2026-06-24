"""
Evaluate a trained A2C model for Role B on DroneDispatch-v0.

This script loads a saved Actor-Critic model and evaluates it with the
same metric system used by the simulator baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make repository root visible.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from networks import ActorCritic, flatten_dispatch_obs

try:
    import drone_dispatch_env  # registers env ids
    from drone_dispatch_env import Config
    from drone_dispatch_env.evaluate import evaluate
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. Make sure the simulator files "
        "are included in the repository and the environment is active."
    ) from e


class A2CPolicy:
    """
    Policy wrapper for the saved A2C Actor-Critic model.

    The evaluator expects a policy object with:
        act(obs) -> action
    """

    def __init__(self, weights_path: Path, cfg: Config):
        self.cfg = cfg
        self.device = torch.device("cpu")

        try:
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(weights_path, map_location=self.device)

        self.model = ActorCritic(
            obs_dim=int(checkpoint["obs_dim"]),
            n_actions=int(checkpoint["n_actions"]),
            hidden_size=int(checkpoint.get("config", {}).get("hidden_size", 128)),
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def act(self, obs):
        obs_vec = flatten_dispatch_obs(obs)
        action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

        obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(obs_tensor)
            masked_logits = logits.masked_fill(mask_tensor <= 0, -1e9)
            action = torch.argmax(masked_logits, dim=-1)

        return int(action.item())


def parse_seeds(seed_text: str):
    return [int(s) for s in seed_text.split(",") if s.strip() != ""]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/role_b/a2c_seed0.pt",
        help="Path to saved A2C weights.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_standard.yaml",
        help="Evaluation config path.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated evaluation seeds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/role_b/a2c_eval_seed0.json",
        help="Where to save evaluation results.",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.seeds)

    policy = A2CPolicy(ROOT / args.weights, cfg)
    results = evaluate(policy, cfg, seeds)

    print(json.dumps(results["mean"], indent=2))

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[info] Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()