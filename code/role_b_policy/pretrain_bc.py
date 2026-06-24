"""
Behavior Cloning pretraining for Role B.

This script uses greedy_nearest as an expert policy and trains the actor part
of the Actor-Critic network to imitate the expert's actions.

The saved model can later be used as a warm-start for A2C fine-tuning.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from networks import ActorCritic, flatten_dispatch_obs
from manual_optim import ManualAdam

try:
    import drone_dispatch_env
    from drone_dispatch_env import Config, DroneDispatchEnv
    from drone_dispatch_env.baselines import GreedyNearest
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. Make sure the simulator files "
        "are included in the repository and the environment is active."
    ) from e


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collect_expert_data(cfg: Config, seed: int, num_samples: int):
    """
    Collect state-action pairs from the greedy_nearest expert policy.
    """
    env = DroneDispatchEnv(cfg)
    expert = GreedyNearest(cfg)

    obs, _ = env.reset(seed=seed)

    obs_list = []
    mask_list = []
    action_list = []

    episode = 0

    while len(action_list) < num_samples:
        obs_vec = flatten_dispatch_obs(obs)
        action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
        expert_action = int(expert.act(obs))

        obs_list.append(obs_vec)
        mask_list.append(action_mask)
        action_list.append(expert_action)

        obs, reward, terminated, truncated, _ = env.step(expert_action)
        done = bool(terminated or truncated)

        if done:
            episode += 1
            obs, _ = env.reset(seed=seed + episode + 1)

    return (
        np.asarray(obs_list, dtype=np.float32),
        np.asarray(mask_list, dtype=np.float32),
        np.asarray(action_list, dtype=np.int64),
    )


def train_bc(config_path: Path) -> None:
    cfg_data = load_yaml(config_path)

    seed = int(cfg_data.get("seed", 0))
    num_samples = int(cfg_data.get("num_samples", 10000))
    epochs = int(cfg_data.get("epochs", 8))
    batch_size = int(cfg_data.get("batch_size", 256))
    learning_rate = float(cfg_data.get("learning_rate", 3e-4))
    hidden_size = int(cfg_data.get("hidden_size", 128))
    max_grad_norm = float(cfg_data.get("max_grad_norm", 0.5))
    run_name = str(cfg_data.get("run_name", "bc_greedy"))

    eval_cfg_path = cfg_data.get("eval", {}).get("config_path", "configs/eval_standard.yaml")
    env_cfg = Config.from_yaml(str(ROOT / eval_cfg_path))

    set_seed(seed)
    device = torch.device("cpu")

    print("[info] Collecting greedy_nearest expert data")
    print(f"[info] num_samples={num_samples}, seed={seed}")

    obs_data, mask_data, action_data = collect_expert_data(env_cfg, seed, num_samples)

    obs_dim = obs_data.shape[1]
    n_actions = env_cfg.n_actions

    model = ActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hidden_size,
    ).to(device)

    optimizer = ManualAdam(model.parameters(), lr=learning_rate)

    log_path = ROOT / "logs" / "role_b" / f"{run_name}_seed{seed}.csv"
    weights_path = ROOT / "weights" / "role_b" / f"{run_name}_seed{seed}.pt"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "loss", "accuracy"],
        )
        writer.writeheader()

        print("[info] Starting behavior cloning training")
        print(f"[info] obs_dim={obs_dim}, n_actions={n_actions}, epochs={epochs}")

        for epoch in range(1, epochs + 1):
            indices = rng.permutation(num_samples)

            total_loss = 0.0
            total_correct = 0
            total_seen = 0

            for start in range(0, num_samples, batch_size):
                batch_idx = indices[start:start + batch_size]

                obs_batch = torch.tensor(obs_data[batch_idx], dtype=torch.float32, device=device)
                mask_batch = torch.tensor(mask_data[batch_idx], dtype=torch.float32, device=device)
                action_batch = torch.tensor(action_data[batch_idx], dtype=torch.long, device=device)

                logits, _ = model(obs_batch)
                masked_logits = logits.masked_fill(mask_batch <= 0, -1e9)

                loss = F.cross_entropy(masked_logits, action_batch)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                with torch.no_grad():
                    pred = torch.argmax(masked_logits, dim=-1)
                    total_correct += int((pred == action_batch).sum().item())
                    total_seen += int(action_batch.numel())
                    total_loss += float(loss.item()) * int(action_batch.numel())

            mean_loss = total_loss / max(total_seen, 1)
            accuracy = total_correct / max(total_seen, 1)

            writer.writerow(
                {
                    "epoch": epoch,
                    "loss": mean_loss,
                    "accuracy": accuracy,
                }
            )
            f.flush()

            print(f"[epoch] {epoch}/{epochs} loss={mean_loss:.4f} accuracy={accuracy:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "seed": seed,
            "config": cfg_data,
            "source": "greedy_nearest_behavior_cloning",
        },
        weights_path,
    )

    print(f"[info] Saved BC weights to {weights_path}")
    print(f"[info] Saved BC logs to {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/role_b/bc_pretrain.yaml",
        help="Path to BC pretraining config.",
    )
    args = parser.parse_args()

    train_bc(ROOT / args.config)


if __name__ == "__main__":
    main()