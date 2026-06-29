from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class OfflinePolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x):
        return self.net(x)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_sources(text: str) -> List[str]:
    if text.strip() == "":
        return []
    return [s.strip() for s in text.split(",") if s.strip()]


def compute_trajectory_returns(rewards: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """
    Assign total episode return to each transition in that episode.
    Dataset was collected source-by-source and seed-by-seed, so done=True
    marks the end of each collected trajectory.
    """
    traj_returns = np.zeros_like(rewards, dtype=np.float32)

    start = 0
    n = len(rewards)

    for i in range(n):
        if dones[i] > 0.5 or i == n - 1:
            total_return = float(np.sum(rewards[start : i + 1]))
            traj_returns[start : i + 1] = total_return
            start = i + 1

    return traj_returns


def masked_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(masks <= 0, -1e9)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/offline_mixed_dataset.npz",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="weights/offline_rl_iql/iql_lite.pt",
    )

    parser.add_argument(
        "--log-output",
        type=str,
        default="weights/offline_rl_iql/iql_lite_training_log.csv",
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="greedy_nearest,milp_rolling,role_c_meta_dynaq_v2",
        help="Comma-separated source names used for IQL-lite training.",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Advantage/return weight strength.",
    )

    parser.add_argument(
        "--max-weight",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cpu")

    dataset_path = ROOT / args.dataset
    data = np.load(dataset_path, allow_pickle=True)

    states = data["states"].astype(np.float32)
    actions = data["actions"].astype(np.int64)
    masks = data["action_masks"].astype(np.float32)
    rewards = data["rewards"].astype(np.float32)
    dones = data["dones"].astype(np.float32)
    sources = data["sources"].astype(str)

    selected_sources = parse_sources(args.sources)

    if selected_sources:
        source_mask = np.isin(sources, selected_sources)
    else:
        source_mask = np.ones(len(actions), dtype=bool)

    traj_returns = compute_trajectory_returns(rewards, dones)

    states = states[source_mask]
    actions = actions[source_mask]
    masks = masks[source_mask]
    selected_source_array = sources[source_mask]
    selected_returns = traj_returns[source_mask]

    if len(actions) == 0:
        raise RuntimeError("No transitions selected. Check --sources argument.")

    mean_return = float(np.mean(selected_returns))
    std_return = float(np.std(selected_returns) + 1e-6)

    normalized_returns = (selected_returns - mean_return) / std_return
    sample_weights = np.exp(args.beta * normalized_returns)
    sample_weights = np.clip(sample_weights, args.min_weight, args.max_weight)
    sample_weights = sample_weights / (np.mean(sample_weights) + 1e-8)

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)

    n = int(actions_t.shape[0])
    state_dim = int(states_t.shape[1])
    action_dim = int(masks_t.shape[1])

    model = OfflinePolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=args.hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = ROOT / args.output
    log_path = ROOT / args.log_output

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("IQL-lite / Advantage-Weighted BC training")
    print(f"Dataset path        : {dataset_path}")
    print(f"Selected sources    : {selected_sources}")
    print(f"Selected transitions: {n}")
    print("Source counts:")
    unique_sources, counts = np.unique(selected_source_array, return_counts=True)
    for s, c in zip(unique_sources, counts):
        print(f"  {s}: {c}")

    print(f"Mean trajectory return: {mean_return:.3f}")
    print(f"Std trajectory return : {std_return:.3f}")
    print(f"Weight mean/min/max   : {sample_weights.mean():.3f} / {sample_weights.min():.3f} / {sample_weights.max():.3f}")

    rows = []

    for step in range(1, args.steps + 1):
        idx = torch.randint(0, n, (args.batch_size,), device=device)

        batch_states = states_t[idx]
        batch_actions = actions_t[idx]
        batch_masks = masks_t[idx]
        batch_weights = weights_t[idx]

        logits = model(batch_states)
        logits = masked_logits(logits, batch_masks)

        ce = F.cross_entropy(logits, batch_actions, reduction="none")
        loss = (ce * batch_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == batch_actions).float().mean().item()
                weighted_acc = (
                    ((pred == batch_actions).float() * batch_weights).sum()
                    / (batch_weights.sum() + 1e-8)
                ).item()

            row = {
                "step": step,
                "loss": float(loss.item()),
                "accuracy": acc,
                "weighted_accuracy": weighted_acc,
            }

            rows.append(row)

            print(
                f"[IQL-lite] step={step:5d} "
                f"loss={loss.item():.4f} "
                f"acc={acc:.3f} "
                f"weighted_acc={weighted_acc:.3f}"
            )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "accuracy", "weighted_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    torch.save(
        {
            "model_type": "iql_lite_advantage_weighted_bc",
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "hidden_size": args.hidden_size,
            "dataset_path": str(dataset_path),
            "config": {
                "sources": selected_sources,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "beta": args.beta,
                "max_weight": args.max_weight,
                "min_weight": args.min_weight,
                "mean_trajectory_return": mean_return,
                "std_trajectory_return": std_return,
            },
        },
        output_path,
    )

    print("\nIQL-lite training finished")
    print(f"Model saved to: {output_path}")
    print(f"Log saved to  : {log_path}")


if __name__ == "__main__":
    main()