from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLE_A_DIR = ROOT / "code" / "role_a_dqn"
ROLE_B_DIR = ROOT / "code" / "role_b_policy"
ROLE_C_DIR = ROOT / "code" / "role_c_planning"

for p in [ROLE_A_DIR, ROLE_B_DIR, ROLE_C_DIR]:
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn

from drone_dispatch_env import Config, DroneDispatchEnv
from drone_dispatch_env.baselines import GreedyNearest, RandomPolicy, MILPRolling


# ------------------------------------------------------------
# Observation processing
# ------------------------------------------------------------

def flatten_full_obs(obs: Dict[str, Any]) -> np.ndarray:
    drones = np.asarray(obs["drones"], dtype=np.float32).flatten()
    orders = np.asarray(obs["orders"], dtype=np.float32).flatten()
    grid = np.asarray(obs["grid"], dtype=np.float32).flatten()
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    return np.concatenate([drones, orders, grid, time]).astype(np.float32)


def valid_random_action(obs: Dict[str, Any]) -> int:
    mask = np.asarray(obs["action_mask"], dtype=np.float32)
    valid = np.flatnonzero(mask > 0)

    if len(valid) == 0:
        return 0

    return int(np.random.choice(valid))


def sanitize_action(obs: Dict[str, Any], action: int) -> int:
    mask = np.asarray(obs["action_mask"], dtype=np.float32)

    if 0 <= int(action) < len(mask) and mask[int(action)] > 0:
        return int(action)

    return valid_random_action(obs)


# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------

def make_env(cfg: Config, seed: int):
    """
    Robust environment creation helper.
    This tries the common constructor forms used by the simulator.
    """
    constructors = [
        lambda: DroneDispatchEnv(cfg, seed=seed),
        lambda: DroneDispatchEnv(config=cfg, seed=seed),
        lambda: DroneDispatchEnv(cfg),
        lambda: DroneDispatchEnv(config=cfg),
    ]

    last_error = None

    for fn in constructors:
        try:
            return fn()
        except Exception as e:
            last_error = e

    raise RuntimeError(f"Could not create DroneDispatchEnv. Last error: {last_error}")


def reset_env(env, seed: int):
    try:
        out = env.reset(seed=seed)
    except TypeError:
        out = env.reset()

    if isinstance(out, tuple):
        return out[0]

    return out


def step_env(env, action: int):
    out = env.step(action)

    if len(out) == 5:
        next_obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return next_obs, float(reward), done, info

    if len(out) == 4:
        next_obs, reward, done, info = out
        return next_obs, float(reward), bool(done), info

    raise RuntimeError("Unexpected env.step output format.")


# ------------------------------------------------------------
# Policy wrappers
# ------------------------------------------------------------

class SimpleDQNNetwork(nn.Module):
    """
    Compatible with Role A DQN and Double DQN weights:
    Linear(581,256) -> ReLU -> Linear(256,256) -> ReLU -> Linear(256,169)
    """

    def __init__(self, state_size: int, action_size: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
        )

    def forward(self, x):
        return self.net(x)


class RoleADQNPolicy:
    def __init__(self, weights_path: Path, state_size: int, action_size: int):
        self.device = torch.device("cpu")
        self.model = SimpleDQNNetwork(state_size, action_size).to(self.device)

        state_dict = torch.load(weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def act(self, obs):
        state = flatten_full_obs(obs)
        mask = np.asarray(obs["action_mask"], dtype=np.float32)

        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(x).cpu().numpy()[0]

        q_values[mask <= 0] = -1e9
        return int(np.argmax(q_values))


class SafePolicyWrapper:
    """
    Prevents one weak/buggy optional policy from breaking the dataset collection.
    If the wrapped policy fails or returns an invalid action, a valid random action is used.
    """

    def __init__(self, name: str, policy):
        self.name = name
        self.policy = policy
        self.failures = 0

    def act(self, obs):
        try:
            action = int(self.policy.act(obs))
            return sanitize_action(obs, action)
        except Exception:
            self.failures += 1
            return valid_random_action(obs)


def try_make_baseline_policy(name: str, cfg: Config):
    policy_classes = {
        "random": RandomPolicy,
        "greedy_nearest": GreedyNearest,
        "milp_rolling": MILPRolling,
    }

    cls = policy_classes[name]

    for args in [(cfg,), tuple()]:
        try:
            return SafePolicyWrapper(name, cls(*args))
        except Exception:
            pass

    raise RuntimeError(f"Could not initialize baseline policy: {name}")


def build_policy_sources(cfg: Config, include_milp: bool = False):
    sources = []

    # Baselines
    for name in ["random", "greedy_nearest"]:
        try:
            sources.append(try_make_baseline_policy(name, cfg))
            print(f"[ok] loaded policy source: {name}")
        except Exception as e:
            print(f"[skip] {name}: {e}")

    if include_milp:
        try:
            sources.append(try_make_baseline_policy("milp_rolling", cfg))
            print("[ok] loaded policy source: milp_rolling")
        except Exception as e:
            print(f"[skip] milp_rolling: {e}")

    # Role A DQN
    try:
        dqn_path = ROOT / "weights" / "dqn_seed0.pt"
        if dqn_path.exists():
            policy = RoleADQNPolicy(
                weights_path=dqn_path,
                state_size=581,
                action_size=169,
            )
            sources.append(SafePolicyWrapper("role_a_dqn", policy))
            print("[ok] loaded policy source: role_a_dqn")
        else:
            print("[skip] role_a_dqn: weights/dqn_seed0.pt not found")
    except Exception as e:
        print(f"[skip] role_a_dqn: {e}")

    # Role A Double DQN
    try:
        double_path = ROOT / "weights" / "double_dqn_seed0.pt"
        if double_path.exists():
            policy = RoleADQNPolicy(
                weights_path=double_path,
                state_size=581,
                action_size=169,
            )
            sources.append(SafePolicyWrapper("role_a_double_dqn", policy))
            print("[ok] loaded policy source: role_a_double_dqn")
        else:
            print("[skip] role_a_double_dqn: weights/double_dqn_seed0.pt not found")
    except Exception as e:
        print(f"[skip] role_a_double_dqn: {e}")

    # Role B A2C
    try:
        from eval_a2c import A2CPolicy

        a2c_path = ROOT / "weights" / "role_b" / "a2c_seed0.pt"
        if a2c_path.exists():
            policy = A2CPolicy(a2c_path, cfg)
            sources.append(SafePolicyWrapper("role_b_a2c", policy))
            print("[ok] loaded policy source: role_b_a2c")
        else:
            print("[skip] role_b_a2c: weights/role_b/a2c_seed0.pt not found")
    except Exception as e:
        print(f"[skip] role_b_a2c: {e}")

    # Role B Behavior Cloning
    try:
        from eval_a2c import A2CPolicy

        bc_path = ROOT / "weights" / "role_b" / "bc_greedy_seed0.pt"
        if bc_path.exists():
            policy = A2CPolicy(bc_path, cfg)
            sources.append(SafePolicyWrapper("role_b_bc", policy))
            print("[ok] loaded policy source: role_b_bc")
        else:
            print("[skip] role_b_bc: weights/role_b/bc_greedy_seed0.pt not found")
    except Exception as e:
        print(f"[skip] role_b_bc: {e}")

    # Role C final planner
    try:
        from meta_dynaq_planner import TrainedMetaDynaQPlanner

        role_c_path = ROOT / "weights" / "meta_dynaq_v2_seed0.pkl.gz"
        if role_c_path.exists():
            policy = TrainedMetaDynaQPlanner(weights_path=str(role_c_path))
            sources.append(SafePolicyWrapper("role_c_meta_dynaq_v2", policy))
            print("[ok] loaded policy source: role_c_meta_dynaq_v2")
        else:
            print("[skip] role_c_meta_dynaq_v2: weights/meta_dynaq_v2_seed0.pkl.gz not found")
    except Exception as e:
        print(f"[skip] role_c_meta_dynaq_v2: {e}")

    if len(sources) == 0:
        raise RuntimeError("No policy sources could be loaded.")

    return sources


# ------------------------------------------------------------
# Dataset collection
# ------------------------------------------------------------

def collect_from_policy(
    cfg: Config,
    policy_wrapper: SafePolicyWrapper,
    seed: int,
    max_steps: int,
):
    env = make_env(cfg, seed=seed)
    obs = reset_env(env, seed=seed)

    states = []
    actions = []
    rewards = []
    next_states = []
    dones = []
    masks = []
    next_masks = []
    sources = []

    total_reward = 0.0
    steps = 0

    while steps < max_steps:
        state_vec = flatten_full_obs(obs)
        mask = np.asarray(obs["action_mask"], dtype=np.float32)

        action = policy_wrapper.act(obs)
        action = sanitize_action(obs, action)

        next_obs, reward, done, info = step_env(env, action)

        next_state_vec = flatten_full_obs(next_obs)
        next_mask = np.asarray(next_obs["action_mask"], dtype=np.float32)

        states.append(state_vec)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state_vec)
        dones.append(float(done))
        masks.append(mask)
        next_masks.append(next_mask)
        sources.append(policy_wrapper.name)

        total_reward += reward
        steps += 1

        obs = next_obs

        if done:
            break

    try:
        env.close()
    except Exception:
        pass

    summary = {
        "source": policy_wrapper.name,
        "seed": seed,
        "steps": steps,
        "return": total_reward,
        "fallback_actions": policy_wrapper.failures,
    }

    data = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "dones": dones,
        "action_masks": masks,
        "next_action_masks": next_masks,
        "sources": sources,
    }

    return data, summary


def parse_seeds(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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
        "--max-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="datasets/offline_mixed_dataset.npz",
    )

    parser.add_argument(
        "--summary-output",
        type=str,
        default="logs/offline_dataset_summary.csv",
    )

    parser.add_argument(
        "--include-milp",
        action="store_true",
        help="Include MILP Rolling in dataset collection. This may be slower.",
    )

    args = parser.parse_args()

    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.seeds)

    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    policies = build_policy_sources(
        cfg=cfg,
        include_milp=args.include_milp,
    )

    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    all_masks = []
    all_next_masks = []
    all_sources = []
    summaries = []

    for policy in policies:
        for seed in seeds:
            print(f"[collect] source={policy.name}, seed={seed}")

            data, summary = collect_from_policy(
                cfg=cfg,
                policy_wrapper=policy,
                seed=seed,
                max_steps=args.max_steps,
            )

            all_states.extend(data["states"])
            all_actions.extend(data["actions"])
            all_rewards.extend(data["rewards"])
            all_next_states.extend(data["next_states"])
            all_dones.extend(data["dones"])
            all_masks.extend(data["action_masks"])
            all_next_masks.extend(data["next_action_masks"])
            all_sources.extend(data["sources"])

            summaries.append(summary)

            print(
                f"  steps={summary['steps']} "
                f"return={summary['return']:.2f} "
                f"fallback_actions={summary['fallback_actions']}"
            )

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        states=np.asarray(all_states, dtype=np.float32),
        actions=np.asarray(all_actions, dtype=np.int64),
        rewards=np.asarray(all_rewards, dtype=np.float32),
        next_states=np.asarray(all_next_states, dtype=np.float32),
        dones=np.asarray(all_dones, dtype=np.float32),
        action_masks=np.asarray(all_masks, dtype=np.float32),
        next_action_masks=np.asarray(all_next_masks, dtype=np.float32),
        sources=np.asarray(all_sources),
    )

    summary_path = ROOT / args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "seed", "steps", "return", "fallback_actions"],
        )
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)

    print("\nDATASET COLLECTION FINISHED")
    print(f"Dataset saved to : {output_path}")
    print(f"Summary saved to : {summary_path}")
    print(f"Transitions      : {len(all_actions)}")
    print(f"Sources          : {sorted(set(all_sources))}")


if __name__ == "__main__":
    main()