from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from drone_dispatch_env import Config, DroneDispatchEnv
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_seeds(text: str) -> List[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def make_env(cfg: Config, seed: int):
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

    raise RuntimeError(f"Could not create environment. Last error: {last_error}")


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


def make_baseline(cls, cfg):
    for args in [(cfg,), tuple()]:
        try:
            return cls(*args)
        except Exception:
            pass

    raise RuntimeError(f"Could not initialize baseline: {cls.__name__}")


def get_mean_metrics(result):
    if isinstance(result, dict) and "mean" in result:
        return result["mean"]
    return result


# ------------------------------------------------------------
# Action mapping
# ------------------------------------------------------------

def infer_dims(obs: Dict[str, Any]) -> Tuple[int, int, int]:
    n_drones = int(np.asarray(obs["drones"]).shape[0])
    k_orders = int(np.asarray(obs["orders"]).shape[0])
    action_size = int(np.asarray(obs["action_mask"]).shape[0])
    return n_drones, k_orders, action_size


def env_action_from_local(
    drone_id: int,
    local_action: int,
    n_drones: int,
    k_orders: int,
) -> int:
    """
    Local actions:
    0 ... k_orders-1  -> assign this drone to visible order slot
    k_orders          -> send this drone to charger
    k_orders + 1      -> noop
    """
    if local_action < k_orders:
        return drone_id * k_orders + local_action

    if local_action == k_orders:
        return n_drones * k_orders + drone_id

    return n_drones * k_orders + n_drones


def local_action_mask(
    obs: Dict[str, Any],
    drone_id: int,
    n_drones: int,
    k_orders: int,
) -> np.ndarray:
    env_mask = np.asarray(obs["action_mask"], dtype=np.float32)
    local_dim = k_orders + 2

    mask = np.zeros(local_dim, dtype=np.float32)

    for local_action in range(local_dim):
        env_action = env_action_from_local(
            drone_id=drone_id,
            local_action=local_action,
            n_drones=n_drones,
            k_orders=k_orders,
        )

        if 0 <= env_action < len(env_mask) and env_mask[env_action] > 0:
            mask[local_action] = 1.0

    if mask.sum() <= 0:
        mask[-1] = 1.0

    return mask


def sanitize_env_action(obs: Dict[str, Any], action: int) -> int:
    env_mask = np.asarray(obs["action_mask"], dtype=np.float32)

    if 0 <= int(action) < len(env_mask) and env_mask[int(action)] > 0:
        return int(action)

    valid = np.flatnonzero(env_mask > 0)

    if len(valid) == 0:
        return 0

    return int(valid[-1])


# ------------------------------------------------------------
# Local observation
# ------------------------------------------------------------

def local_observation(
    obs: Dict[str, Any],
    drone_id: int,
    n_drones: int,
) -> np.ndarray:
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    own_drone = drones[drone_id].flatten()
    all_orders = orders.flatten()
    drone_id_feature = np.asarray([drone_id / max(1, n_drones - 1)], dtype=np.float32)

    return np.concatenate(
        [
            own_drone,
            all_orders,
            time,
            drone_id_feature,
        ]
    ).astype(np.float32)


# ------------------------------------------------------------
# Neural network and replay buffer
# ------------------------------------------------------------

class SharedIDQN(nn.Module):
    def __init__(self, obs_dim: int, local_action_dim: int, hidden_size: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, local_action_dim),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.data = []
        self.pos = 0

    def push(self, transition):
        if len(self.data) < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.pos] = transition

        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)

        states, actions, rewards, next_states, dones, masks, next_masks = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(masks, dtype=np.float32),
            np.asarray(next_masks, dtype=np.float32),
        )

    def __len__(self):
        return len(self.data)


# ------------------------------------------------------------
# Policy
# ------------------------------------------------------------

class IDQNPolicy:
    def __init__(
        self,
        model: SharedIDQN,
        obs_dim: int,
        local_action_dim: int,
        epsilon: float = 0.0,
    ):
        self.model = model
        self.obs_dim = obs_dim
        self.local_action_dim = local_action_dim
        self.epsilon = float(epsilon)
        self.device = torch.device("cpu")

        self.model.to(self.device)
        self.model.eval()

    def select_proposal(
        self,
        obs: Dict[str, Any],
        drone_id: int,
        n_drones: int,
        k_orders: int,
    ) -> Tuple[int, int, float]:
        local_obs = local_observation(obs, drone_id, n_drones)
        mask = local_action_mask(obs, drone_id, n_drones, k_orders)

        valid_local = np.flatnonzero(mask > 0)

        if len(valid_local) == 0:
            local_action = k_orders + 1
            env_action = env_action_from_local(drone_id, local_action, n_drones, k_orders)
            return local_action, env_action, -1e9

        if random.random() < self.epsilon:
            local_action = int(np.random.choice(valid_local))
            env_action = env_action_from_local(drone_id, local_action, n_drones, k_orders)
            return local_action, env_action, 0.0

        x = torch.tensor(local_obs, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.model(x).cpu().numpy()[0]

        q_values[mask <= 0] = -1e9

        local_action = int(np.argmax(q_values))
        env_action = env_action_from_local(drone_id, local_action, n_drones, k_orders)

        return local_action, env_action, float(q_values[local_action])

    def act(self, obs: Dict[str, Any]) -> int:
        n_drones, k_orders, _ = infer_dims(obs)

        proposals = []

        for drone_id in range(n_drones):
            local_action, env_action, score = self.select_proposal(
                obs=obs,
                drone_id=drone_id,
                n_drones=n_drones,
                k_orders=k_orders,
            )

            env_action = sanitize_env_action(obs, env_action)

            proposals.append(
                {
                    "drone_id": drone_id,
                    "local_action": local_action,
                    "env_action": env_action,
                    "score": score,
                }
            )

        best = max(proposals, key=lambda p: p["score"])

        return sanitize_env_action(obs, best["env_action"])

    def act_with_training_info(self, obs: Dict[str, Any]):
        n_drones, k_orders, _ = infer_dims(obs)

        proposals = []

        for drone_id in range(n_drones):
            local_action, env_action, score = self.select_proposal(
                obs=obs,
                drone_id=drone_id,
                n_drones=n_drones,
                k_orders=k_orders,
            )

            env_action = sanitize_env_action(obs, env_action)

            proposals.append(
                {
                    "drone_id": drone_id,
                    "local_action": local_action,
                    "env_action": env_action,
                    "score": score,
                }
            )

        best = max(proposals, key=lambda p: p["score"])

        best["env_action"] = sanitize_env_action(obs, best["env_action"])

        return best


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_idqn(args):
    set_seed(args.train_seed)

    cfg = Config.from_yaml(str(ROOT / args.config))

    env = make_env(cfg, seed=args.train_seed)
    obs = reset_env(env, seed=args.train_seed)

    n_drones, k_orders, _ = infer_dims(obs)
    local_obs_dim = int(local_observation(obs, 0, n_drones).shape[0])
    local_action_dim = int(k_orders + 2)

    model = SharedIDQN(
        obs_dim=local_obs_dim,
        local_action_dim=local_action_dim,
        hidden_size=args.hidden_size,
    )

    target_model = SharedIDQN(
        obs_dim=local_obs_dim,
        local_action_dim=local_action_dim,
        hidden_size=args.hidden_size,
    )

    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    replay = ReplayBuffer(args.replay_size)

    log_rows = []

    epsilon = args.epsilon_start

    print("Training parameter-shared IDQN-lite")
    print(f"Local obs dim     : {local_obs_dim}")
    print(f"Local action dim  : {local_action_dim}")
    print(f"Drones            : {n_drones}")
    print(f"Visible orders    : {k_orders}")

    device = torch.device("cpu")
    model.to(device)
    target_model.to(device)

    global_step = 0

    for ep in range(1, args.episodes + 1):
        obs = reset_env(env, seed=args.train_seed + ep)

        episode_return = 0.0
        episode_steps = 0
        done = False

        while not done and episode_steps < args.max_steps:
            model.eval()

            policy = IDQNPolicy(
                model=model,
                obs_dim=local_obs_dim,
                local_action_dim=local_action_dim,
                epsilon=epsilon,
            )

            chosen = policy.act_with_training_info(obs)

            drone_id = int(chosen["drone_id"])
            local_action = int(chosen["local_action"])
            env_action = int(chosen["env_action"])

            state = local_observation(obs, drone_id, n_drones)
            mask = local_action_mask(obs, drone_id, n_drones, k_orders)

            next_obs, reward, done, info = step_env(env, env_action)

            next_state = local_observation(next_obs, drone_id, n_drones)
            next_mask = local_action_mask(next_obs, drone_id, n_drones, k_orders)

            replay.push(
                (
                    state,
                    local_action,
                    reward,
                    next_state,
                    float(done),
                    mask,
                    next_mask,
                )
            )

            obs = next_obs
            episode_return += reward
            episode_steps += 1
            global_step += 1

            if len(replay) >= args.batch_size:
                model.train()

                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                    masks,
                    next_masks,
                ) = replay.sample(args.batch_size)

                states_t = torch.tensor(states, dtype=torch.float32, device=device)
                actions_t = torch.tensor(actions, dtype=torch.long, device=device)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)
                next_masks_t = torch.tensor(next_masks, dtype=torch.float32, device=device)

                q_values = model(states_t)
                q_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_model(next_states_t)
                    next_q = next_q.masked_fill(next_masks_t <= 0, -1e9)
                    next_q_max = next_q.max(dim=1).values

                    target = rewards_t + args.gamma * (1.0 - dones_t) * next_q_max

                loss = F.mse_loss(q_action, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if global_step % args.target_update_every == 0:
                    target_model.load_state_dict(model.state_dict())

            epsilon = max(args.epsilon_min, epsilon * args.epsilon_decay)

        log_rows.append(
            {
                "episode": ep,
                "episode_return": episode_return,
                "steps": episode_steps,
                "epsilon": epsilon,
                "replay_size": len(replay),
            }
        )

        if ep == 1 or ep % args.print_every == 0:
            print(
                f"ep={ep:4d} "
                f"return={episode_return:9.2f} "
                f"steps={episode_steps:4d} "
                f"epsilon={epsilon:.3f} "
                f"replay={len(replay)}"
            )

    try:
        env.close()
    except Exception:
        pass

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_type": "parameter_shared_idqn_lite",
            "model_state_dict": model.state_dict(),
            "obs_dim": local_obs_dim,
            "local_action_dim": local_action_dim,
            "hidden_size": args.hidden_size,
            "config": {
                "episodes": args.episodes,
                "gamma": args.gamma,
                "lr": args.lr,
                "epsilon_start": args.epsilon_start,
                "epsilon_min": args.epsilon_min,
                "epsilon_decay": args.epsilon_decay,
                "train_seed": args.train_seed,
            },
        },
        output_path,
    )

    log_path = ROOT / args.log
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "episode_return",
                "steps",
                "epsilon",
                "replay_size",
            ],
        )
        writer.writeheader()
        writer.writerows(log_rows)

    print("\nIDQN-lite training finished")
    print(f"Weights saved to: {output_path}")
    print(f"Log saved to    : {log_path}")

    return output_path


# ------------------------------------------------------------
# Load and evaluate
# ------------------------------------------------------------

def load_idqn_policy(weights_path: Path) -> IDQNPolicy:
    device = torch.device("cpu")

    try:
        checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights_path, map_location=device)

    obs_dim = int(checkpoint["obs_dim"])
    local_action_dim = int(checkpoint["local_action_dim"])
    hidden_size = int(checkpoint.get("hidden_size", 128))

    model = SharedIDQN(
        obs_dim=obs_dim,
        local_action_dim=local_action_dim,
        hidden_size=hidden_size,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return IDQNPolicy(
        model=model,
        obs_dim=obs_dim,
        local_action_dim=local_action_dim,
        epsilon=0.0,
    )


def evaluate_idqn(args, weights_path: Path):
    cfg = Config.from_yaml(str(ROOT / args.config))
    seeds = parse_seeds(args.eval_seeds)

    results = {}

    print("\nEvaluating Multi-Agent IDQN-lite")
    print(f"Weights: {weights_path}")
    print(f"Seeds  : {seeds}")

    idqn_policy = load_idqn_policy(weights_path)
    idqn_metrics = evaluate(idqn_policy, cfg, seeds)
    results["Parameter-shared IDQN-lite"] = idqn_metrics

    print(json.dumps(idqn_metrics, indent=2))

    if args.with_baselines:
        baselines = [
            ("Random Policy", RandomPolicy),
            ("Greedy Nearest", GreedyNearest),
            ("MILP Rolling", MILPRolling),
        ]

        for name, cls in baselines:
            print(f"\nEvaluating baseline: {name}")

            try:
                policy = make_baseline(cls, cfg)
                metrics = evaluate(policy, cfg, seeds)
                results[name] = metrics
                print(json.dumps(metrics, indent=2))
            except Exception as e:
                print(f"[skip] {name}: {e}")

    print("\nMULTI-AGENT EVALUATION COMPARISON")
    print("-" * 112)
    print(
        f"{'Policy':<30} | {'Cost/Order':>10} | {'Success':>8} | "
        f"{'On-time':>8} | {'Delivered':>10} | {'Dropped':>8} | "
        f"{'Depletion':>10} | {'Return':>10}"
    )
    print("-" * 112)

    for name, result in results.items():
        m = get_mean_metrics(result)

        print(
            f"{name:<30} | "
            f"{m.get('cost_per_order', 0):>10.4f} | "
            f"{m.get('success_rate', 0):>8.3f} | "
            f"{m.get('ontime_rate', 0):>8.3f} | "
            f"{m.get('n_delivered', 0):>10.2f} | "
            f"{m.get('n_dropped', 0):>8.2f} | "
            f"{m.get('depletion_events', 0):>10.2f} | "
            f"{m.get('episode_return', 0):>10.2f}"
        )

    print("-" * 112)

    eval_path = ROOT / args.eval_output
    eval_path.parent.mkdir(parents=True, exist_ok=True)

    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nEvaluation saved to: {eval_path}")


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
        "--train-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--eval-seeds",
        type=str,
        default="0,1,2",
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
    )

    parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--replay-size",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--epsilon-min",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
    )

    parser.add_argument(
        "--target-update-every",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=str,
        default="weights/multi_agent/idqn_seed0.pt",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="logs/multi_agent/idqn_train_seed0.csv",
    )

    parser.add_argument(
        "--eval-output",
        type=str,
        default="logs/multi_agent/idqn_eval_seed0.json",
    )

    parser.add_argument(
        "--eval-only",
        action="store_true",
    )

    parser.add_argument(
        "--with-baselines",
        action="store_true",
    )

    args = parser.parse_args()

    weights_path = ROOT / args.output

    if not args.eval_only:
        weights_path = train_idqn(args)

    evaluate_idqn(args, weights_path)


if __name__ == "__main__":
    main()