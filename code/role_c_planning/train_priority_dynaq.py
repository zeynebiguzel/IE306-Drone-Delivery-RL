from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import yaml

# Repository root: your-repo/
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import drone_dispatch_env
    from drone_dispatch_env import Config, DroneDispatchEnv
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. "
        "Run `pip install -e .` from the repository root."
    ) from e

from network import StateEncoder
from priority_dynaq import PriorityDynaQAgent


def load_yaml(path: Path) -> dict:
    if path is None:
        return {}

    if not path.is_absolute():
        path = ROOT / path

    if not path.exists():
        print(f"[warning] Config not found: {path}")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_env_config(train_cfg: dict) -> Config:
    eval_cfg_path = train_cfg.get(
        "eval_config",
        train_cfg.get("config_path", "configs/eval_standard.yaml"),
    )

    full_path = ROOT / eval_cfg_path

    if full_path.exists():
        return Config.from_yaml(str(full_path))

    print(f"[warning] Could not find {full_path}. Using default Config().")
    return Config()


def make_paths(run_name: str, seed: int):
    weights_dir = ROOT / "weights"
    logs_dir = ROOT / "logs"

    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    weights_path = weights_dir / f"{run_name}_seed{seed}.pkl.gz"
    log_path = logs_dir / f"{run_name}_seed{seed}.csv"

    return weights_path, log_path


def make_agent(train_cfg: dict, action_dim: int) -> PriorityDynaQAgent:
    encoder = StateEncoder(
        position_bin=int(train_cfg.get("position_bin", 2)),
        order_position_bin=int(train_cfg.get("order_position_bin", 3)),
        soc_bins=int(train_cfg.get("soc_bins", 5)),
        age_bin=int(train_cfg.get("age_bin", 10)),
        time_bins=int(train_cfg.get("time_bins", 10)),
        top_k_orders=int(train_cfg.get("top_k_orders", 8)),
    )

    alpha = float(train_cfg.get("alpha", 0.10))

    epsilon_start = float(
        train_cfg.get("epsilon_start", 1.0)
    )

    epsilon_min = float(
        train_cfg.get(
            "epsilon_min",
            train_cfg.get("epsilon", 0.05)
        )
    )

    agent = PriorityDynaQAgent(
        action_dim=action_dim,
        alpha=alpha,
        gamma=float(train_cfg.get("gamma", 0.99)),
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=float(train_cfg.get("epsilon_decay", 0.995)),
        planning_steps=int(train_cfg.get("planning_steps", 20)),
        priority_threshold=float(train_cfg.get("priority_threshold", 1e-4)),
        encoder=encoder,
    )

    return agent

def get_episode_stats(env) -> dict:
    stats = getattr(env, "stats", {})

    delivered = int(stats.get("delivered", 0))
    dropped = int(stats.get("dropped", 0))
    depletion_events = int(stats.get("depletion_events", 0))
    energy = float(stats.get("energy", 0.0))
    late_cost = float(stats.get("late_cost", 0.0))
    drop_cost = float(stats.get("drop_cost", 0.0))
    depletion_cost = float(stats.get("depletion_cost", 0.0))

    denom = max(delivered, 1)

    cost_per_order = (
        energy
        + late_cost
        + drop_cost
        + depletion_cost
    ) / denom

    total_orders = delivered + dropped

    success_rate = (
        delivered / total_orders
        if total_orders > 0
        else 0.0
    )

    return {
        "delivered": delivered,
        "dropped": dropped,
        "depletion_events": depletion_events,
        "energy": energy,
        "cost_per_order": cost_per_order,
        "success_rate": success_rate,
    }


def train_priority_dynaq(config_path: Path, seed: int) -> None:
    train_cfg = load_yaml(config_path)

    set_seed(seed)

    run_name = str(train_cfg.get("run_name", "priority_dynaq"))

    num_episodes = int(train_cfg.get("num_episodes", 300))
    max_decision_steps = int(train_cfg.get("max_decision_steps", 5000))

    env_cfg = load_env_config(train_cfg)
    env = DroneDispatchEnv(env_cfg)

    obs, info = env.reset(seed=seed)

    action_dim = int(len(obs["action_mask"]))

    print("ACTION SIZE:", action_dim)
    print("ROLE C METHOD: Tabular Priority Dyna-Q")
    print("PLANNING STEPS:", train_cfg.get("planning_steps", 20))

    agent = make_agent(
        train_cfg=train_cfg,
        action_dim=action_dim,
    )

    weights_path, log_path = make_paths(
        run_name=run_name,
        seed=seed,
    )

    best_cost = float("inf")
    best_reward = -float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_return",
                "decision_steps",
                "epsilon",
                "mean_real_td_error",
                "mean_planning_td_error",
                "planning_updates",
                "model_size",
                "queue_size",
                "delivered",
                "dropped",
                "depletion_events",
                "energy",
                "cost_per_order",
                "success_rate",
            ],
        )

        writer.writeheader()

        for episode in range(1, num_episodes + 1):
            obs, info = env.reset(seed=seed + episode)

            terminated = False
            truncated = False

            episode_return = 0.0
            decision_steps = 0

            real_td_errors = []
            planning_td_errors = []
            planning_updates = 0

            while not terminated and not truncated:
                action = agent.select_action(
                    obs,
                    training=True,
                )

                next_obs, reward, terminated, truncated, info = env.step(
                    action
                )

                done = bool(terminated or truncated)

                update_info = agent.observe(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=done,
                )

                real_td_errors.append(
                    abs(float(update_info["real_td_error"]))
                )

                planning_td_errors.append(
                    float(update_info["mean_planning_td_error"])
                )

                planning_updates += int(
                    update_info["planning_updates"]
                )

                episode_return += float(reward)
                decision_steps += 1

                obs = next_obs

                if decision_steps >= max_decision_steps:
                    break

            agent.decay_epsilon()

            stats = get_episode_stats(env)

            mean_real_td = (
                float(np.mean(real_td_errors))
                if real_td_errors
                else 0.0
            )

            mean_planning_td = (
                float(np.mean(planning_td_errors))
                if planning_td_errors
                else 0.0
            )

            row = {
                "episode": episode,
                "episode_return": episode_return,
                "decision_steps": decision_steps,
                "epsilon": agent.epsilon,
                "mean_real_td_error": mean_real_td,
                "mean_planning_td_error": mean_planning_td,
                "planning_updates": planning_updates,
                "model_size": len(agent.model),
                "queue_size": len(agent.priority_queue),
                "delivered": stats["delivered"],
                "dropped": stats["dropped"],
                "depletion_events": stats["depletion_events"],
                "energy": stats["energy"],
                "cost_per_order": stats["cost_per_order"],
                "success_rate": stats["success_rate"],
            }

            writer.writerow(row)
            f.flush()

            print(
                f"Episode {episode:04d}/{num_episodes} | "
                f"Return={episode_return:8.2f} | "
                f"Cost={stats['cost_per_order']:.4f} | "
                f"Success={stats['success_rate']:.3f} | "
                f"Delivered={stats['delivered']} | "
                f"Dropped={stats['dropped']} | "
                f"Eps={agent.epsilon:.3f} | "
                f"Model={len(agent.model)} | "
                f"PlanUpdates={planning_updates}"
            )

            improved_cost = (
                stats["delivered"] > 0
                and stats["cost_per_order"] < best_cost
            )

            improved_reward = episode_return > best_reward

            if improved_cost or improved_reward:
                best_cost = min(best_cost, stats["cost_per_order"])
                best_reward = max(best_reward, episode_return)
                agent.save(str(weights_path))

        agent.save(str(weights_path))

    print("\nTRAINING FINISHED")
    print(f"MODEL SAVED: {weights_path}")
    print(f"LOG SAVED: {log_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="configs/priority_dynaq.yaml",
        help="Path to Priority Dyna-Q config file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed.",
    )

    args = parser.parse_args()

    train_priority_dynaq(
        config_path=Path(args.config),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()