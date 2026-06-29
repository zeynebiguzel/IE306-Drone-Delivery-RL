from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]
MA_DIR = ROOT / "code" / "multi_agent"
ROLE_C_DIR = ROOT / "code" / "role_c_planning"

for p in [ROOT, MA_DIR, ROLE_C_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


from drone_dispatch_env import Config
from drone_dispatch_env.baselines import GreedyNearest, MILPRolling

from run_idqn import (
    SharedIDQN,
    make_env,
    reset_env,
    step_env,
    infer_dims,
    local_observation,
    local_action_mask,
    sanitize_env_action,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_seeds(text: str) -> List[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def parse_sources(text: str) -> List[str]:
    return [s.strip() for s in text.split(",") if s.strip()]


def make_baseline_policy(cls, cfg):
    for args in [(cfg,), tuple()]:
        try:
            return cls(*args)
        except Exception:
            pass

    raise RuntimeError(f"Could not initialize {cls.__name__}")


def decode_env_action(
    action: int,
    n_drones: int,
    k_orders: int,
) -> Tuple[str, int | None, int]:
    """
    Decode centralized env action into local multi-agent label.

    Returns:
        action_type, selected_drone, local_action

    local_action:
        0 ... k_orders-1  -> order slot
        k_orders          -> charge
        k_orders + 1      -> noop
    """
    assign_limit = n_drones * k_orders
    charge_limit = assign_limit + n_drones

    if action < assign_limit:
        drone_id = int(action // k_orders)
        order_slot = int(action % k_orders)
        return "assign", drone_id, order_slot

    if action < charge_limit:
        drone_id = int(action - assign_limit)
        return "charge", drone_id, k_orders

    return "noop", None, k_orders + 1


def build_expert_sources(cfg: Config, source_names: List[str]):
    experts = []

    for source_name in source_names:
        if source_name == "greedy_nearest":
            try:
                experts.append((source_name, make_baseline_policy(GreedyNearest, cfg)))
                print(f"[ok] loaded expert: {source_name}")
            except Exception as e:
                print(f"[skip] {source_name}: {e}")

        elif source_name == "milp_rolling":
            try:
                experts.append((source_name, make_baseline_policy(MILPRolling, cfg)))
                print(f"[ok] loaded expert: {source_name}")
            except Exception as e:
                print(f"[skip] {source_name}: {e}")

        elif source_name == "role_c_meta_dynaq_v2":
            try:
                from meta_dynaq_planner import TrainedMetaDynaQPlanner

                weights_path = ROOT / "weights" / "meta_dynaq_v2_seed0.pkl.gz"
                expert = TrainedMetaDynaQPlanner(weights_path=str(weights_path))

                experts.append((source_name, expert))
                print(f"[ok] loaded expert: {source_name}")

            except Exception as e:
                print(f"[skip] {source_name}: {e}")

        else:
            print(f"[skip] unknown expert source: {source_name}")

    if len(experts) == 0:
        raise RuntimeError("No expert sources were loaded.")

    return experts


# ------------------------------------------------------------
# Dataset collection
# ------------------------------------------------------------

def collect_bc_samples(
    cfg: Config,
    expert_name: str,
    expert_policy,
    seed: int,
    max_steps: int,
):
    env = make_env(cfg, seed=seed)
    obs = reset_env(env, seed=seed)

    n_drones, k_orders, _ = infer_dims(obs)

    local_states = []
    local_actions = []
    local_masks = []
    weights = []
    sources = []

    total_return = 0.0
    steps = 0

    while steps < max_steps:
        try:
            expert_action = int(expert_policy.act(obs))
        except Exception:
            expert_action = 0

        expert_action = sanitize_env_action(obs, expert_action)

        action_type, selected_drone, selected_local_action = decode_env_action(
            action=expert_action,
            n_drones=n_drones,
            k_orders=k_orders,
        )

        # Convert one centralized expert action into per-drone local labels.
        for drone_id in range(n_drones):
            local_obs = local_observation(obs, drone_id, n_drones)
            mask = local_action_mask(obs, drone_id, n_drones, k_orders)

            noop_action = k_orders + 1

            if action_type in {"assign", "charge"} and drone_id == selected_drone:
                label = selected_local_action
                sample_weight = 2.0
            elif action_type == "noop":
                label = noop_action
                sample_weight = 0.4
            else:
                label = noop_action
                sample_weight = 0.15

            # If selected label is invalid for this local state, fall back to noop.
            if not (0 <= label < len(mask)) or mask[label] <= 0:
                if 0 <= noop_action < len(mask) and mask[noop_action] > 0:
                    label = noop_action
                else:
                    valid = np.flatnonzero(mask > 0)
                    if len(valid) == 0:
                        continue
                    label = int(valid[0])

            local_states.append(local_obs)
            local_actions.append(label)
            local_masks.append(mask)
            weights.append(sample_weight)
            sources.append(expert_name)

        next_obs, reward, done, info = step_env(env, expert_action)

        total_return += reward
        steps += 1
        obs = next_obs

        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    return {
        "states": local_states,
        "actions": local_actions,
        "masks": local_masks,
        "weights": weights,
        "sources": sources,
        "steps": steps,
        "return": total_return,
    }


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_standard.yaml",
    )

    parser.add_argument(
        "--sources",
        type=str,
        default="greedy_nearest,milp_rolling,role_c_meta_dynaq_v2",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--train-steps",
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
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
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
        "--output",
        type=str,
        default="weights/multi_agent/idqn_bc_seed0.pt",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="logs/multi_agent/idqn_bc_train_seed0.csv",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    cfg = Config.from_yaml(str(ROOT / args.config))
    source_names = parse_sources(args.sources)
    seeds = parse_seeds(args.seeds)

    experts = build_expert_sources(cfg, source_names)

    all_states = []
    all_actions = []
    all_masks = []
    all_weights = []
    summary_rows = []

    print("\nCollecting multi-agent BC samples")

    for expert_name, expert_policy in experts:
        for seed in seeds:
            print(f"[collect] expert={expert_name}, seed={seed}")

            data = collect_bc_samples(
                cfg=cfg,
                expert_name=expert_name,
                expert_policy=expert_policy,
                seed=seed,
                max_steps=args.max_steps,
            )

            all_states.extend(data["states"])
            all_actions.extend(data["actions"])
            all_masks.extend(data["masks"])
            all_weights.extend(data["weights"])

            summary_rows.append(
                {
                    "expert": expert_name,
                    "seed": seed,
                    "steps": data["steps"],
                    "return": data["return"],
                    "samples": len(data["actions"]),
                }
            )

            print(
                f"  steps={data['steps']} "
                f"return={data['return']:.2f} "
                f"samples={len(data['actions'])}"
            )

    if len(all_actions) == 0:
        raise RuntimeError("No BC samples collected.")

    states = torch.tensor(np.asarray(all_states, dtype=np.float32), dtype=torch.float32)
    actions = torch.tensor(np.asarray(all_actions, dtype=np.int64), dtype=torch.long)
    masks = torch.tensor(np.asarray(all_masks, dtype=np.float32), dtype=torch.float32)
    weights = torch.tensor(np.asarray(all_weights, dtype=np.float32), dtype=torch.float32)

    weights = weights / (weights.mean() + 1e-8)

    n = int(actions.shape[0])
    obs_dim = int(states.shape[1])
    local_action_dim = int(masks.shape[1])

    model = SharedIDQN(
        obs_dim=obs_dim,
        local_action_dim=local_action_dim,
        hidden_size=args.hidden_size,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    output_path = ROOT / args.output
    log_path = ROOT / args.log

    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("\nTraining multi-agent BC warm-start")
    print(f"Samples          : {n}")
    print(f"Local obs dim    : {obs_dim}")
    print(f"Local action dim : {local_action_dim}")

    rows = []

    for step in range(1, args.train_steps + 1):
        idx = torch.randint(0, n, (args.batch_size,))

        batch_states = states[idx]
        batch_actions = actions[idx]
        batch_masks = masks[idx]
        batch_weights = weights[idx]

        logits = model(batch_states)
        logits = logits.masked_fill(batch_masks <= 0, -1e9)

        loss_per_sample = F.cross_entropy(
            logits,
            batch_actions,
            reduction="none",
        )

        loss = (loss_per_sample * batch_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step == 1 or step % args.log_every == 0:
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = (pred == batch_actions).float().mean().item()
                weighted_acc = (
                    ((pred == batch_actions).float() * batch_weights).sum()
                    / (batch_weights.sum() + 1e-8)
                ).item()

            rows.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "accuracy": acc,
                    "weighted_accuracy": weighted_acc,
                }
            )

            print(
                f"[MA-BC] step={step:5d} "
                f"loss={loss.item():.4f} "
                f"acc={acc:.3f} "
                f"weighted_acc={weighted_acc:.3f}"
            )

    torch.save(
        {
            "model_type": "parameter_shared_idqn_bc_warmstart",
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "local_action_dim": local_action_dim,
            "hidden_size": args.hidden_size,
            "config": {
                "sources": source_names,
                "seeds": seeds,
                "train_steps": args.train_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
            },
        },
        output_path,
    )

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "loss", "accuracy", "weighted_accuracy"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = log_path.with_name("idqn_bc_collection_summary.csv")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["expert", "seed", "steps", "return", "samples"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nMulti-agent BC warm-start finished")
    print(f"Weights saved to : {output_path}")
    print(f"Train log saved  : {log_path}")
    print(f"Summary saved    : {summary_path}")


if __name__ == "__main__":
    main()
    