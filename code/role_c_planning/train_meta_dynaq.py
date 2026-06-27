from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


try:
    import drone_dispatch_env
    from drone_dispatch_env import Config, DroneDispatchEnv
    from drone_dispatch_env.baselines import GreedyNearest
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. "
        "Run `pip install -e .` from the repository root."
    ) from e


from meta_dynaq_planner import (
    META_KEEP,
    META_SWAP,
    META_CHARGE,
    META_ACTION_NAMES,
    MetaDynaQAgent,
    MetaStateEncoder,
    meta_to_env_action,
    meta_action_mask,
    _greedy_assignment_context,
)


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


def call_greedy(greedy, obs):
    candidates = []

    if hasattr(greedy, "act"):
        candidates.append(greedy.act)

    if hasattr(greedy, "select_action"):
        candidates.append(greedy.select_action)

    if hasattr(greedy, "predict"):
        candidates.append(greedy.predict)

    if callable(greedy):
        candidates.append(greedy)

    for fn in candidates:
        for args in [
            (obs,),
            (obs, None),
            (obs, {}),
        ]:
            try:
                out = fn(*args)

                if isinstance(out, tuple):
                    out = out[0]

                return int(out)

            except TypeError:
                continue

            except Exception:
                continue

    action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
    valid_actions = np.flatnonzero(action_mask)

    if len(valid_actions) == 0:
        return 0

    return int(valid_actions[0])


def sanitize_env_action(obs, action: int) -> int:
    action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
    valid_actions = np.flatnonzero(action_mask)

    if len(valid_actions) == 0:
        return 0

    action = int(action)

    if 0 <= action < len(action_mask) and action_mask[action] > 0:
        return action

    return int(valid_actions[0])


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


def get_depletion_events(env) -> int:
    stats = getattr(env, "stats", {})
    return int(stats.get("depletion_events", 0))


def make_agent(train_cfg: dict) -> MetaDynaQAgent:
    encoder = MetaStateEncoder(
        soc_bins=int(train_cfg.get("soc_bins", 10)),
        trip_bin=int(train_cfg.get("trip_bin", 5)),
        age_bin=int(train_cfg.get("age_bin", 10)),
        time_bins=int(train_cfg.get("time_bins", 10)),
    )

    agent = MetaDynaQAgent(
        alpha=float(train_cfg.get("alpha", 0.12)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        epsilon_start=float(train_cfg.get("epsilon_start", 1.0)),
        epsilon_min=float(train_cfg.get("epsilon_min", 0.02)),
        epsilon_decay=float(train_cfg.get("epsilon_decay", 0.992)),
        planning_steps=int(train_cfg.get("planning_steps", 30)),
        priority_threshold=float(train_cfg.get("priority_threshold", 1e-4)),
        encoder=encoder,
    )

    return agent
def shaped_training_reward(
    raw_reward: float,
    obs,
    greedy_action: int,
    meta_action: int,
    reason: str,
    depletion_before: int,
    depletion_after: int,
) -> float:
    """
    Training-only reward shaping V2.

    Goal:
        - Keep greedy when it is clearly safe.
        - Encourage Dyna-Q to intervene more in near-risk states.
        - Strongly penalize depletion.
    """

    reward = float(raw_reward)

    ctx = _greedy_assignment_context(
        obs=obs,
        greedy_action=greedy_action,
    )

    risk_flag = int(ctx["risk_flag"])
    risk_margin = float(ctx["risk_margin"])
    soc = float(ctx["soc"])

    near_risk = (
        risk_flag == 1
        or risk_margin < 0.08
        or soc < 0.32
    )

    real_risk = (
        risk_flag == 1
        or risk_margin < 0.03
        or soc < 0.24
    )

    severe_risk = (
        soc < 0.16
        or risk_margin < -0.04
    )

    depletion_increased = depletion_after > depletion_before

    if depletion_increased:
        reward -= 10.0

    if meta_action == META_KEEP:
        if severe_risk:
            reward -= 1.20
        elif real_risk:
            reward -= 0.70
        elif near_risk:
            reward -= 0.20
        else:
            reward += 0.05

    elif meta_action == META_SWAP:
        if reason == "safe_swap":
            if severe_risk:
                reward += 1.60
            elif real_risk:
                reward += 1.20
            elif near_risk:
                reward += 0.50
            else:
                reward -= 0.30
        else:
            reward -= 1.00

    elif meta_action == META_CHARGE:
        if reason == "charge_risky":
            if severe_risk:
                reward += 1.10
            elif real_risk:
                reward += 0.30
            else:
                reward -= 0.60
        else:
            reward -= 1.00

    return float(reward)


def train_meta_dynaq(config_path: Path, seed: int) -> None:
    train_cfg = load_yaml(config_path)

    set_seed(seed)

    run_name = str(train_cfg.get("run_name", "meta_dynaq"))

    num_episodes = int(train_cfg.get("num_episodes", 250))
    max_decision_steps = int(train_cfg.get("max_decision_steps", 5000))

    env_cfg = load_env_config(train_cfg)
    env = DroneDispatchEnv(env_cfg)
    greedy = GreedyNearest(env_cfg)

    agent = make_agent(train_cfg)

    weights_path, log_path = make_paths(
        run_name=run_name,
        seed=seed,
    )

    print("ROLE C METHOD: Meta Dyna-Q Battery-Reassignment Planner")
    print("META ACTION SIZE: 3")
    print("META ACTIONS:")
    for k, v in META_ACTION_NAMES.items():
        print(f"  {k}: {v}")
    print("PLANNING STEPS:", agent.planning_steps)
    print("WEIGHTS:", weights_path)
    print("LOG:", log_path)

    best_cost = float("inf")
    best_return = -float("inf")

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "raw_episode_return",
                "shaped_episode_return",
                "decision_steps",
                "epsilon",
                "mean_real_td_error",
                "mean_planning_td_error",
                "planning_updates",
                "model_size",
                "queue_size",
                "meta_keep",
                "meta_swap",
                "meta_charge",
                "meta_fallback",
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

            raw_episode_return = 0.0
            shaped_episode_return = 0.0
            decision_steps = 0

            real_td_errors = []
            planning_td_errors = []
            planning_updates = 0

            meta_counts = {
                "meta_keep": 0,
                "meta_swap": 0,
                "meta_charge": 0,
                "meta_fallback": 0,
            }

            while not terminated and not truncated:
                greedy_action = call_greedy(greedy, obs)
                greedy_action = sanitize_env_action(obs, greedy_action)

                meta_action = agent.select_meta_action(
                    obs=obs,
                    greedy_action=greedy_action,
                    training=True,
                )

                env_action, reason = meta_to_env_action(
                    obs=obs,
                    greedy_action=greedy_action,
                    meta_action=meta_action,
                )

                env_action = sanitize_env_action(obs, env_action)

                if reason == "keep_greedy":
                    meta_counts["meta_keep"] += 1
                elif reason == "safe_swap":
                    meta_counts["meta_swap"] += 1
                elif reason == "charge_risky":
                    meta_counts["meta_charge"] += 1
                else:
                    meta_counts["meta_fallback"] += 1

                depletion_before = get_depletion_events(env)

                next_obs, raw_reward, terminated, truncated, info = env.step(
                    env_action
                )

                done = bool(terminated or truncated)

                depletion_after = get_depletion_events(env)

                shaped_reward = shaped_training_reward(
                    raw_reward=float(raw_reward),
                    obs=obs,
                    greedy_action=greedy_action,
                    meta_action=meta_action,
                    reason=reason,
                    depletion_before=depletion_before,
                    depletion_after=depletion_after,
                )

                next_greedy_action = call_greedy(greedy, next_obs)
                next_greedy_action = sanitize_env_action(
                    next_obs,
                    next_greedy_action,
                )

                update_info = agent.observe(
                    obs=obs,
                    greedy_action=greedy_action,
                    meta_action=meta_action,
                    reward=shaped_reward,
                    next_obs=next_obs,
                    next_greedy_action=next_greedy_action,
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

                raw_episode_return += float(raw_reward)
                shaped_episode_return += float(shaped_reward)
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
                "raw_episode_return": raw_episode_return,
                "shaped_episode_return": shaped_episode_return,
                "decision_steps": decision_steps,
                "epsilon": agent.epsilon,
                "mean_real_td_error": mean_real_td,
                "mean_planning_td_error": mean_planning_td,
                "planning_updates": planning_updates,
                "model_size": len(agent.model),
                "queue_size": len(agent.priority_queue),
                "meta_keep": meta_counts["meta_keep"],
                "meta_swap": meta_counts["meta_swap"],
                "meta_charge": meta_counts["meta_charge"],
                "meta_fallback": meta_counts["meta_fallback"],
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
                f"RawReturn={raw_episode_return:8.2f} | "
                f"Shaped={shaped_episode_return:8.2f} | "
                f"Cost={stats['cost_per_order']:.4f} | "
                f"Success={stats['success_rate']:.3f} | "
                f"Delivered={stats['delivered']} | "
                f"Dropped={stats['dropped']} | "
                f"Depl={stats['depletion_events']} | "
                f"Eps={agent.epsilon:.3f} | "
                f"Keep={meta_counts['meta_keep']} | "
                f"Swap={meta_counts['meta_swap']} | "
                f"Charge={meta_counts['meta_charge']} | "
                f"Model={len(agent.model)} | "
                f"PlanUpdates={planning_updates}"
            )

            improved_cost = (
                stats["delivered"] > 0
                and stats["cost_per_order"] < best_cost
            )

            improved_return = raw_episode_return > best_return

            if improved_cost or improved_return:
                best_cost = min(best_cost, stats["cost_per_order"])
                best_return = max(best_return, raw_episode_return)
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
        default="configs/meta_dynaq.yaml",
        help="Path to Meta Dyna-Q config file.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Training seed.",
    )

    args = parser.parse_args()

    train_meta_dynaq(
        config_path=Path(args.config),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()