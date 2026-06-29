from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


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
# Dataset
# ------------------------------------------------------------

class OfflineDataset:
    def __init__(self, path: Path, device: torch.device):
        data = np.load(path, allow_pickle=True)

        self.states = torch.tensor(data["states"], dtype=torch.float32, device=device)
        self.actions = torch.tensor(data["actions"], dtype=torch.long, device=device)
        self.rewards = torch.tensor(data["rewards"], dtype=torch.float32, device=device)
        self.next_states = torch.tensor(data["next_states"], dtype=torch.float32, device=device)
        self.dones = torch.tensor(data["dones"], dtype=torch.float32, device=device)
        self.action_masks = torch.tensor(data["action_masks"], dtype=torch.float32, device=device)
        self.next_action_masks = torch.tensor(data["next_action_masks"], dtype=torch.float32, device=device)

        self.n = int(self.actions.shape[0])
        self.state_dim = int(self.states.shape[1])
        self.action_dim = int(self.action_masks.shape[1])

        self.sources = data["sources"]

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idx = torch.randint(0, self.n, (batch_size,), device=self.states.device)

        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_states": self.next_states[idx],
            "dones": self.dones[idx],
            "action_masks": self.action_masks[idx],
            "next_action_masks": self.next_action_masks[idx],
        }

    def source_counts(self):
        unique, counts = np.unique(self.sources, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def masked_q_values(q_values: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    return q_values.masked_fill(action_mask <= 0, -1e9)


def hard_update(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict())


def compute_q_diagnostics(
    model: OfflineQNetwork,
    dataset: OfflineDataset,
    batch_size: int,
) -> Tuple[float, float, float]:
    batch = dataset.sample(batch_size)

    with torch.no_grad():
        q_values = model(batch["states"])
        masked_q = masked_q_values(q_values, batch["action_masks"])

        q_dataset = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)
        q_max = masked_q.max(dim=1).values
        ood_gap = q_max - q_dataset

    return (
        float(q_dataset.mean().item()),
        float(q_max.mean().item()),
        float(ood_gap.mean().item()),
    )


def save_checkpoint(
    path: Path,
    model: OfflineQNetwork,
    model_type: str,
    dataset_path: Path,
    config: Dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_type": model_type,
            "model_state_dict": model.state_dict(),
            "state_dim": config["state_dim"],
            "action_dim": config["action_dim"],
            "hidden_size": config["hidden_size"],
            "dataset_path": str(dataset_path),
            "config": config,
        },
        path,
    )


# ------------------------------------------------------------
# Behavior Cloning
# ------------------------------------------------------------

def train_behavior_cloning(
    dataset: OfflineDataset,
    output_dir: Path,
    dataset_path: Path,
    args,
):
    print("\nTraining Behavior Cloning model")

    model = OfflineQNetwork(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_size=args.hidden_size,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.bc_lr)

    log_path = output_dir / "bc_training_log.csv"
    rows = []

    for step in range(1, args.bc_steps + 1):
        batch = dataset.sample(args.batch_size)

        logits = model(batch["states"])
        masked_logits = masked_q_values(logits, batch["action_masks"])

        loss = F.cross_entropy(masked_logits, batch["actions"])

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            with torch.no_grad():
                pred = masked_logits.argmax(dim=1)
                acc = (pred == batch["actions"]).float().mean().item()

            q_dataset, q_max, ood_gap = compute_q_diagnostics(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
            )

            row = {
                "step": step,
                "loss": float(loss.item()),
                "accuracy": acc,
                "q_dataset_mean": q_dataset,
                "q_max_mean": q_max,
                "ood_gap": ood_gap,
            }
            rows.append(row)

            print(
                f"[BC] step={step:5d} "
                f"loss={loss.item():.4f} "
                f"acc={acc:.3f} "
                f"ood_gap={ood_gap:.3f}"
            )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "accuracy",
                "q_dataset_mean",
                "q_max_mean",
                "ood_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = output_dir / "bc_policy.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        model_type="behavior_cloning",
        dataset_path=dataset_path,
        config={
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "hidden_size": args.hidden_size,
            "bc_lr": args.bc_lr,
            "bc_steps": args.bc_steps,
            "batch_size": args.batch_size,
        },
    )

    print(f"[BC] saved to {checkpoint_path}")


# ------------------------------------------------------------
# Offline DQN and CQL-lite
# ------------------------------------------------------------

def train_offline_dqn(
    dataset: OfflineDataset,
    output_dir: Path,
    dataset_path: Path,
    args,
    model_type: str,
    cql_alpha: float,
):
    assert model_type in {"naive_offline_dqn", "cql_lite"}

    print(f"\nTraining {model_type}")

    model = OfflineQNetwork(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_size=args.hidden_size,
    ).to(args.device)

    target_model = OfflineQNetwork(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        hidden_size=args.hidden_size,
    ).to(args.device)

    hard_update(target_model, model)
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.dqn_lr)

    log_path = output_dir / f"{model_type}_training_log.csv"
    rows = []

    for step in range(1, args.dqn_steps + 1):
        batch = dataset.sample(args.batch_size)

        q_values = model(batch["states"])
        q_action = q_values.gather(1, batch["actions"].unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = target_model(batch["next_states"])
            masked_next_q = masked_q_values(next_q_values, batch["next_action_masks"])
            next_q_max = masked_next_q.max(dim=1).values

            target = (
                args.reward_scale * batch["rewards"]
                + args.gamma * (1.0 - batch["dones"]) * next_q_max
            )

        bellman_loss = F.mse_loss(q_action, target)

        if model_type == "cql_lite":
            masked_current_q = masked_q_values(q_values, batch["action_masks"])
            conservative_penalty = torch.logsumexp(masked_current_q, dim=1).mean() - q_action.mean()
            loss = bellman_loss + cql_alpha * conservative_penalty
        else:
            conservative_penalty = torch.tensor(0.0, device=args.device)
            loss = bellman_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.target_update_every == 0:
            hard_update(target_model, model)

        if step % args.log_every == 0 or step == 1:
            q_dataset, q_max, ood_gap = compute_q_diagnostics(
                model=model,
                dataset=dataset,
                batch_size=args.batch_size,
            )

            row = {
                "step": step,
                "loss": float(loss.item()),
                "bellman_loss": float(bellman_loss.item()),
                "conservative_penalty": float(conservative_penalty.item()),
                "q_dataset_mean": q_dataset,
                "q_max_mean": q_max,
                "ood_gap": ood_gap,
            }

            rows.append(row)

            print(
                f"[{model_type}] step={step:5d} "
                f"loss={loss.item():.4f} "
                f"bellman={bellman_loss.item():.4f} "
                f"cql={conservative_penalty.item():.4f} "
                f"ood_gap={ood_gap:.3f}"
            )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "bellman_loss",
                "conservative_penalty",
                "q_dataset_mean",
                "q_max_mean",
                "ood_gap",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_name = "offline_dqn.pt" if model_type == "naive_offline_dqn" else "cql_lite.pt"
    checkpoint_path = output_dir / checkpoint_name

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        model_type=model_type,
        dataset_path=dataset_path,
        config={
            "state_dim": dataset.state_dim,
            "action_dim": dataset.action_dim,
            "hidden_size": args.hidden_size,
            "dqn_lr": args.dqn_lr,
            "dqn_steps": args.dqn_steps,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "reward_scale": args.reward_scale,
            "target_update_every": args.target_update_every,
            "cql_alpha": cql_alpha,
        },
    )

    print(f"[{model_type}] saved to {checkpoint_path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/offline_mixed_dataset.npz",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="weights/offline_rl",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--bc-steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--dqn-steps",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--bc-lr",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--dqn-lr",
        type=float,
        default=0.0003,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--reward-scale",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--cql-alpha",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--target-update-every",
        type=int,
        default=100,
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

    args = parser.parse_args()

    args.device = torch.device("cpu")

    set_seed(args.seed)

    dataset_path = ROOT / args.dataset
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = OfflineDataset(dataset_path, args.device)

    print("Offline dataset loaded")
    print(f"Dataset path : {dataset_path}")
    print(f"Transitions  : {dataset.n}")
    print(f"State dim    : {dataset.state_dim}")
    print(f"Action dim   : {dataset.action_dim}")
    print("Source counts:")
    for source, count in dataset.source_counts().items():
        print(f"  {source}: {count}")

    train_behavior_cloning(
        dataset=dataset,
        output_dir=output_dir,
        dataset_path=dataset_path,
        args=args,
    )

    train_offline_dqn(
        dataset=dataset,
        output_dir=output_dir,
        dataset_path=dataset_path,
        args=args,
        model_type="naive_offline_dqn",
        cql_alpha=0.0,
    )

    train_offline_dqn(
        dataset=dataset,
        output_dir=output_dir,
        dataset_path=dataset_path,
        args=args,
        model_type="cql_lite",
        cql_alpha=args.cql_alpha,
    )

    print("\nOFFLINE MODEL TRAINING FINISHED")
    print(f"Saved models and logs under: {output_dir}")


if __name__ == "__main__":
    main()