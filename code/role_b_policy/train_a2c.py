"""
Train A2C for Role B on DroneDispatch-v0.

Role B requirement:
- Policy-based / actor-critic methods
- REINFORCE + GAE -> A2C
- DDPG on DroneControl-v0 later

This script implements the A2C part.
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
from torch.distributions import Categorical

# Make sure the repository root is visible when this script is run from anywhere.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from networks import ActorCritic, flatten_dispatch_obs  # local Role B file

try:
    import drone_dispatch_env  # registers env ids
    from drone_dispatch_env import Config, DroneDispatchEnv
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. Make sure the simulator is installed "
        "or included in the repository. Run `pip install -e .` from the simulator root."
    ) from e


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_env_config(train_cfg: Dict[str, Any]) -> Config:
    """
    Load the simulator evaluation config.

    If configs/eval_standard.yaml is not available in the repository yet,
    we fall back to the default Config. This lets the code structure remain
    usable while the team decides how to include the simulator files.
    """
    eval_cfg_path = train_cfg.get("eval", {}).get("config_path", "configs/eval_standard.yaml")
    full_path = ROOT / eval_cfg_path

    if full_path.exists():
        return Config.from_yaml(str(full_path))

    print(f"[warning] Could not find {full_path}. Falling back to default Config().")
    return Config()


def make_log_writer(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "row_type",
            "step",
            "update",
            "episode",
            "episode_return",
            "episode_length",
            "loss",
            "policy_loss",
            "value_loss",
            "entropy",
        ],
    )
    writer.writeheader()
    return f, writer


def train_a2c(config_path: Path) -> None:
    train_cfg = load_yaml(config_path)

    seed = int(train_cfg.get("seed", 0))
    total_timesteps = int(train_cfg.get("total_timesteps", 50_000))
    num_steps = int(train_cfg.get("num_steps", 128))

    gamma = float(train_cfg.get("gamma", 0.99))
    gae_lambda = float(train_cfg.get("gae_lambda", 0.95))
    learning_rate = float(train_cfg.get("learning_rate", 3e-4))
    hidden_size = int(train_cfg.get("hidden_size", 128))

    entropy_coef = float(train_cfg.get("entropy_coef", 0.01))
    value_coef = float(train_cfg.get("value_coef", 0.5))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0.5))
    advantage_normalization = bool(train_cfg.get("advantage_normalization", True))

    set_seed(seed)
    device = torch.device("cpu")

    env_cfg = load_env_config(train_cfg)
    env = DroneDispatchEnv(env_cfg)

    obs, _ = env.reset(seed=seed)
    obs_dim = len(flatten_dispatch_obs(obs))
    n_actions = env_cfg.n_actions

    model = ActorCritic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        hidden_size=hidden_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_path = ROOT / "logs" / "role_b" / f"a2c_seed{seed}.csv"
    weights_path = ROOT / "weights" / "role_b" / f"a2c_seed{seed}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    log_file, writer = make_log_writer(log_path)

    global_step = 0
    update = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0

    print("[info] Starting A2C training")
    print(f"[info] seed={seed}, total_timesteps={total_timesteps}, num_steps={num_steps}")
    print(f"[info] obs_dim={obs_dim}, n_actions={n_actions}")

    try:
        while global_step < total_timesteps:
            update += 1

            obs_list = []
            mask_list = []
            action_list = []
            reward_list = []
            done_list = []
            value_list = []

            for _ in range(num_steps):
                obs_vec = flatten_dispatch_obs(obs)
                action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

                obs_tensor = torch.tensor(obs_vec, dtype=torch.float32, device=device).unsqueeze(0)
                mask_tensor = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

                with torch.no_grad():
                    action, _, _, value = model.get_action_and_value(obs_tensor, mask_tensor)

                action_int = int(action.item())

                next_obs, reward, terminated, truncated, _ = env.step(action_int)
                done = bool(terminated or truncated)

                obs_list.append(obs_vec)
                mask_list.append(action_mask)
                action_list.append(action_int)
                reward_list.append(float(reward))
                done_list.append(float(done))
                value_list.append(float(value.item()))

                global_step += 1
                episode_return += float(reward)
                episode_length += 1

                obs = next_obs

                if done:
                    writer.writerow(
                        {
                            "row_type": "episode",
                            "step": global_step,
                            "update": update,
                            "episode": episode,
                            "episode_return": episode_return,
                            "episode_length": episode_length,
                            "loss": "",
                            "policy_loss": "",
                            "value_loss": "",
                            "entropy": "",
                        }
                    )
                    log_file.flush()

                    print(
                        f"[episode] ep={episode} step={global_step} "
                        f"return={episode_return:.2f} length={episode_length}"
                    )

                    episode += 1
                    episode_return = 0.0
                    episode_length = 0

                    obs, _ = env.reset(seed=seed + episode + 1)

                if global_step >= total_timesteps:
                    break

            # Bootstrap value for the last state if the last transition was not terminal.
            if len(done_list) == 0:
                continue

            if done_list[-1] == 1.0:
                next_value = 0.0
            else:
                next_obs_vec = flatten_dispatch_obs(obs)
                next_obs_tensor = torch.tensor(
                    next_obs_vec, dtype=torch.float32, device=device
                ).unsqueeze(0)
                with torch.no_grad():
                    next_value = float(model.get_value(next_obs_tensor).item())

            rewards = torch.tensor(reward_list, dtype=torch.float32, device=device)
            dones = torch.tensor(done_list, dtype=torch.float32, device=device)
            values = torch.tensor(value_list, dtype=torch.float32, device=device)

            # GAE advantage calculation.
            advantages = torch.zeros_like(rewards, device=device)
            last_gae = 0.0

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_v = torch.tensor(next_value, dtype=torch.float32, device=device)
                else:
                    next_v = values[t + 1]

                next_non_terminal = 1.0 - dones[t]
                delta = rewards[t] + gamma * next_v * next_non_terminal - values[t]
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                advantages[t] = last_gae

            returns = advantages + values

            if advantage_normalization and len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            obs_batch = torch.tensor(np.asarray(obs_list), dtype=torch.float32, device=device)
            mask_batch = torch.tensor(np.asarray(mask_list), dtype=torch.float32, device=device)
            action_batch = torch.tensor(action_list, dtype=torch.long, device=device)

            logits, new_values = model(obs_batch)
            masked_logits = logits.masked_fill(mask_batch <= 0, -1e9)
            dist = Categorical(logits=masked_logits)

            new_log_probs = dist.log_prob(action_batch)
            entropy = dist.entropy().mean()

            policy_loss = -(advantages.detach() * new_log_probs).mean()
            value_loss = F.mse_loss(new_values, returns.detach())

            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            writer.writerow(
                {
                    "row_type": "update",
                    "step": global_step,
                    "update": update,
                    "episode": episode,
                    "episode_return": "",
                    "episode_length": "",
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "value_loss": float(value_loss.item()),
                    "entropy": float(entropy.item()),
                }
            )
            log_file.flush()

            if update % 10 == 0:
                print(
                    f"[update] update={update} step={global_step} "
                    f"loss={loss.item():.4f} policy={policy_loss.item():.4f} "
                    f"value={value_loss.item():.4f} entropy={entropy.item():.4f}"
                )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "obs_dim": obs_dim,
                "n_actions": n_actions,
                "seed": seed,
                "config": train_cfg,
            },
            weights_path,
        )

        print(f"[info] Saved weights to {weights_path}")
        print(f"[info] Saved logs to {log_path}")

    finally:
        log_file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/role_b/a2c.yaml",
        help="Path to Role B A2C config file.",
    )
    args = parser.parse_args()

    train_a2c(ROOT / args.config)


if __name__ == "__main__":
    main()