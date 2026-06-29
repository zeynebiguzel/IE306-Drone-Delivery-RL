from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent

ROLE_A_DIR = ROOT / "code" / "role_a_dqn"
ROLE_B_DIR = ROOT / "code" / "role_b_policy"
ROLE_C_DIR = ROOT / "code" / "role_c_planning"
MULTI_AGENT_DIR = ROOT / "code" / "multi_agent"

for path in [ROOT, ROLE_A_DIR, ROLE_B_DIR, ROLE_C_DIR, MULTI_AGENT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from drone_dispatch_env import Config
from drone_dispatch_env.evaluate import evaluate
from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling


# ------------------------------------------------------------
# General helpers
# ------------------------------------------------------------

def parse_seeds(text: str) -> List[int]:
    return [int(s.strip()) for s in text.split(",") if s.strip()]


def make_baseline(cls, cfg):
    for args in [(cfg,), tuple()]:
        try:
            return cls(*args)
        except Exception:
            pass

    raise RuntimeError(f"Could not initialize baseline: {cls.__name__}")


def get_mean_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    if isinstance(result, dict) and "mean" in result:
        return result["mean"]
    return result


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def safe_evaluate(name: str, policy, cfg: Config, seeds: List[int]):
    print(f"\nEvaluating: {name}")

    try:
        result = evaluate(policy, cfg, seeds)
        print(json.dumps(to_jsonable(result), indent=2))
        return result

    except Exception as e:
        print(f"[FAILED] {name}: {e}")
        return None


def print_comparison_table(results: Dict[str, Optional[Dict[str, Any]]]):
    print("\nFINAL REPRODUCIBILITY COMPARISON")
    print("-" * 132)
    print(
        f"{'Policy':<46} | {'Cost/Order':>10} | {'Success':>8} | "
        f"{'On-time':>8} | {'Delivered':>10} | {'Dropped':>8} | "
        f"{'Depletion':>10} | {'Return':>10}"
    )
    print("-" * 132)

    for name, result in results.items():
        if result is None:
            continue

        m = get_mean_metrics(result)

        print(
            f"{name:<46} | "
            f"{m.get('cost_per_order', 0):>10.4f} | "
            f"{m.get('success_rate', 0):>8.3f} | "
            f"{m.get('ontime_rate', 0):>8.3f} | "
            f"{m.get('n_delivered', 0):>10.2f} | "
            f"{m.get('n_dropped', 0):>8.2f} | "
            f"{m.get('depletion_events', 0):>10.2f} | "
            f"{m.get('episode_return', 0):>10.2f}"
        )

    print("-" * 132)


def save_outputs(
    results: Dict[str, Optional[Dict[str, Any]]],
    output_json: Path,
    output_csv: Path,
):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(results), f, indent=2)

    rows = []

    for name, result in results.items():
        if result is None:
            continue

        m = get_mean_metrics(result)

        rows.append(
            {
                "policy": name,
                "cost_per_order": m.get("cost_per_order", 0),
                "success_rate": m.get("success_rate", 0),
                "ontime_rate": m.get("ontime_rate", 0),
                "mean_delivery_time": m.get("mean_delivery_time", 0),
                "energy_per_order": m.get("energy_per_order", 0),
                "depletion_events": m.get("depletion_events", 0),
                "idle_pct": m.get("idle_pct", 0),
                "charger_utilization": m.get("charger_utilization", 0),
                "n_delivered": m.get("n_delivered", 0),
                "n_dropped": m.get("n_dropped", 0),
                "episode_return": m.get("episode_return", 0),
            }
        )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "policy",
            "cost_per_order",
            "success_rate",
            "ontime_rate",
            "mean_delivery_time",
            "energy_per_order",
            "depletion_events",
            "idle_pct",
            "charger_utilization",
            "n_delivered",
            "n_dropped",
            "episode_return",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON results to: {output_json}")
    print(f"Saved CSV results to : {output_csv}")


# ------------------------------------------------------------
# Observation flattening
# ------------------------------------------------------------

def flatten_obs(obs: Dict[str, Any], expected_dim: Optional[int] = None) -> np.ndarray:
    parts = []

    for key in ["drones", "orders", "grid", "time"]:
        if key in obs:
            parts.append(np.asarray(obs[key], dtype=np.float32).flatten())

    state = np.concatenate(parts).astype(np.float32)

    if expected_dim is not None:
        if len(state) < expected_dim:
            pad = np.zeros(expected_dim - len(state), dtype=np.float32)
            state = np.concatenate([state, pad]).astype(np.float32)
        elif len(state) > expected_dim:
            state = state[:expected_dim].astype(np.float32)

    return state


def safe_action_from_scores(scores: np.ndarray, obs: Dict[str, Any]) -> int:
    mask = np.asarray(obs["action_mask"], dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).flatten()

    if len(scores) < len(mask):
        padded = np.full(len(mask), -1e9, dtype=np.float32)
        padded[: len(scores)] = scores
        scores = padded
    elif len(scores) > len(mask):
        scores = scores[: len(mask)]

    valid = np.flatnonzero(mask > 0)

    if len(valid) == 0:
        return 0

    scores[mask <= 0] = -1e9
    return int(np.argmax(scores))


# ------------------------------------------------------------
# Generic torch checkpoint policy
# Works for most DQN / A2C / BC / IQL style saved networks
# ------------------------------------------------------------

def load_torch_checkpoint(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        possible_keys = [
            "model_state_dict",
            "policy_state_dict",
            "q_network_state_dict",
            "network_state_dict",
            "state_dict",
            "actor_state_dict",
        ]

        for key in possible_keys:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        tensor_items = {
            k: v for k, v in checkpoint.items()
            if torch.is_tensor(v)
        }

        if len(tensor_items) > 0:
            return tensor_items

    raise RuntimeError("Could not extract a torch state_dict from checkpoint.")


def layer_list_from_state_dict(state_dict: Dict[str, torch.Tensor]):
    layers = []

    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue

        if not key.endswith("weight"):
            continue

        if value.ndim != 2:
            continue

        bias_key = key[:-6] + "bias"

        bias = state_dict.get(bias_key, None)

        if bias is not None and torch.is_tensor(bias):
            bias = bias.detach().float()
        else:
            bias = torch.zeros(value.shape[0], dtype=torch.float32)

        layers.append(
            {
                "key": key,
                "weight": value.detach().float(),
                "bias": bias,
            }
        )

    if len(layers) == 0:
        raise RuntimeError("No linear layers found in checkpoint.")

    return layers


def apply_layers(x: torch.Tensor, layers: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    for i, layer in enumerate(layers):
        w = layer["weight"]
        b = layer["bias"]

        x = torch.nn.functional.linear(x, w, b)

        if i < len(layers) - 1:
            x = torch.relu(x)

    return x


class GenericTorchPolicy:
    """
    Dynamic checkpoint reader.

    It supports:
    - plain sequential DQN/BC/IQL networks,
    - dueling DQN checkpoints with value/advantage streams,
    - actor-critic checkpoints with shared layers + actor head.
    """

    def __init__(
        self,
        weights_path: Path,
        prefer_actor: bool = False,
        prefer_dueling: bool = False,
    ):
        self.weights_path = weights_path
        self.checkpoint = load_torch_checkpoint(weights_path)
        self.state_dict = extract_state_dict(self.checkpoint)
        self.layers = layer_list_from_state_dict(self.state_dict)

        self.prefer_actor = prefer_actor
        self.prefer_dueling = prefer_dueling

        self.mode = "sequential"

        self.shared_layers = []
        self.actor_layers = []
        self.value_layers = []
        self.advantage_layers = []
        self.seq_layers = []

        self._build_graph()

        first_layer = self._first_layer()
        self.state_dim = int(first_layer["weight"].shape[1])

    def _first_layer(self):
        for collection in [
            self.shared_layers,
            self.seq_layers,
            self.actor_layers,
            self.advantage_layers,
            self.value_layers,
        ]:
            if len(collection) > 0:
                return collection[0]

        raise RuntimeError("No layers available.")

    def _build_graph(self):
        lower_keys = [layer["key"].lower() for layer in self.layers]

        has_advantage = any(
            ("adv" in k or "advantage" in k) for k in lower_keys
        )
        has_value = any(
            ("value" in k or "val" in k) for k in lower_keys
        )
        has_actor = any(
            ("actor" in k or "policy" in k or "pi" in k) for k in lower_keys
        )
        has_critic = any(
            ("critic" in k or "value" in k or "vf" in k) for k in lower_keys
        )

        if self.prefer_dueling or (has_advantage and has_value):
            self.mode = "dueling"

            for layer in self.layers:
                k = layer["key"].lower()

                if "adv" in k or "advantage" in k:
                    self.advantage_layers.append(layer)
                elif "value" in k or "val" in k:
                    self.value_layers.append(layer)
                else:
                    self.shared_layers.append(layer)

            if len(self.advantage_layers) == 0 or len(self.value_layers) == 0:
                self.mode = "sequential"
                self.seq_layers = self.layers

            return

        if self.prefer_actor or has_actor:
            self.mode = "actor"

            for layer in self.layers:
                k = layer["key"].lower()

                if "actor" in k or "policy" in k or "pi" in k:
                    self.actor_layers.append(layer)
                elif "critic" in k or "value" in k or "vf" in k:
                    continue
                else:
                    self.shared_layers.append(layer)

            if len(self.actor_layers) == 0:
                self.mode = "sequential"
                self.seq_layers = [
                    layer for layer in self.layers
                    if "critic" not in layer["key"].lower()
                    and "value" not in layer["key"].lower()
                    and "vf" not in layer["key"].lower()
                ]

            return

        self.mode = "sequential"
        self.seq_layers = self.layers

    def forward_scores(self, state: np.ndarray) -> np.ndarray:
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if self.mode == "dueling":
                if len(self.shared_layers) > 0:
                    h = apply_layers(x, self.shared_layers)
                else:
                    h = x

                value = apply_layers(h, self.value_layers)
                advantage = apply_layers(h, self.advantage_layers)

                q = value + advantage - advantage.mean(dim=1, keepdim=True)
                out = q

            elif self.mode == "actor":
                if len(self.shared_layers) > 0:
                    h = apply_layers(x, self.shared_layers)
                else:
                    h = x

                out = apply_layers(h, self.actor_layers)

            else:
                out = apply_layers(x, self.seq_layers)

        return out.cpu().numpy()[0]

    def act(self, obs: Dict[str, Any]) -> int:
        state = flatten_obs(obs, expected_dim=self.state_dim)
        scores = self.forward_scores(state)
        return safe_action_from_scores(scores, obs)


# ------------------------------------------------------------
# Role B explicit loader
# ------------------------------------------------------------

def load_role_b_policy(weights_path: Path, cfg: Config):
    """
    First tries Role B's own A2CPolicy class.
    If constructor mismatch happens, falls back to generic actor checkpoint reader.
    """

    try:
        from eval_a2c import A2CPolicy

        constructors = [
            lambda: A2CPolicy(weights_path=str(weights_path)),
            lambda: A2CPolicy(str(weights_path)),
            lambda: A2CPolicy(weights_path),
            lambda: A2CPolicy(cfg, str(weights_path)),
            lambda: A2CPolicy(config=cfg, weights_path=str(weights_path)),
        ]

        last_error = None

        for make_policy in constructors:
            try:
                return make_policy()
            except Exception as e:
                last_error = e

        print(f"[Role B fallback] A2CPolicy constructor failed: {last_error}")

    except Exception as e:
        print(f"[Role B fallback] Could not import A2CPolicy: {e}")

    return GenericTorchPolicy(weights_path, prefer_actor=True)


# ------------------------------------------------------------
# Add policies
# ------------------------------------------------------------

def maybe_add_baselines(
    results: Dict[str, Any],
    cfg: Config,
    seeds: List[int],
):
    specs = [
        ("Random Policy", RandomPolicy),
        ("Greedy Nearest", GreedyNearest),
        ("MILP Rolling", MILPRolling),
    ]

    for name, cls in specs:
        try:
            policy = make_baseline(cls, cfg)
            results[name] = safe_evaluate(name, policy, cfg, seeds)
        except Exception as e:
            print(f"[SKIP] {name}: {e}")


def maybe_add_generic_torch_policy(
    results: Dict[str, Any],
    cfg: Config,
    seeds: List[int],
    name: str,
    weights_path: Path,
    prefer_actor: bool = False,
    prefer_dueling: bool = False,
):
    if not weights_path.exists():
        print(f"[SKIP] {name} weights not found: {weights_path}")
        return

    try:
        policy = GenericTorchPolicy(
            weights_path=weights_path,
            prefer_actor=prefer_actor,
            prefer_dueling=prefer_dueling,
        )

        print(
            f"[LOAD] {name}: {weights_path} "
            f"(mode={policy.mode}, state_dim={policy.state_dim})"
        )

        results[name] = safe_evaluate(name, policy, cfg, seeds)

    except Exception as e:
        print(f"[SKIP] {name} could not be loaded: {e}")


def maybe_add_role_b(
    results: Dict[str, Any],
    cfg: Config,
    seeds: List[int],
    name: str,
    weights_path: Path,
):
    if not weights_path.exists():
        print(f"[SKIP] {name} weights not found: {weights_path}")
        return

    try:
        policy = load_role_b_policy(weights_path, cfg)
        results[name] = safe_evaluate(name, policy, cfg, seeds)

    except Exception as e:
        print(f"[SKIP] {name} could not be loaded: {e}")


def maybe_add_role_c(
    results: Dict[str, Any],
    cfg: Config,
    seeds: List[int],
    weights_path: Path,
):
    if not weights_path.exists():
        print(f"[SKIP] Role C weights not found: {weights_path}")
        return

    try:
        from meta_dynaq_planner import TrainedMetaDynaQPlanner

        policy = TrainedMetaDynaQPlanner(weights_path=str(weights_path))

        results["Role C - Priority Meta Dyna-Q V2"] = safe_evaluate(
            "Role C - Priority Meta Dyna-Q V2",
            policy,
            cfg,
            seeds,
        )

    except Exception as e:
        print(f"[SKIP] Role C could not be loaded: {e}")


def maybe_add_multi_agent(
    results: Dict[str, Any],
    cfg: Config,
    seeds: List[int],
    weights_path: Path,
):
    if not weights_path.exists():
        print(f"[SKIP] Multi-agent weights not found: {weights_path}")
        return

    try:
        from run_idqn import load_idqn_policy

        policy = load_idqn_policy(weights_path)

        results["Multi-Agent - IDQN BC Warm-start"] = safe_evaluate(
            "Multi-Agent - IDQN BC Warm-start",
            policy,
            cfg,
            seeds,
        )

    except Exception as e:
        print(f"[SKIP] Multi-agent could not be loaded: {e}")


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
        "--seeds",
        type=str,
        default="0,1,2",
    )

    # Role A weights
    parser.add_argument(
        "--role-a-dqn-weights",
        type=str,
        default="weights/dqn_seed0.pt",
    )

    parser.add_argument(
        "--role-a-double-weights",
        type=str,
        default="weights/double_dqn_seed0.pt",
    )

    parser.add_argument(
        "--role-a-dueling-weights",
        type=str,
        default="weights/dueling_dqn_seed0.pt",
    )

    # Role B weights
    parser.add_argument(
        "--role-b-a2c-weights",
        type=str,
        default="weights/role_b/a2c_seed0.pt",
    )

    parser.add_argument(
        "--role-b-bc-finetune-weights",
        type=str,
        default="weights/role_b/a2c_bc_finetune_seed0.pt",
    )

    parser.add_argument(
        "--role-b-advnorm-off-weights",
        type=str,
        default="weights/role_b/a2c_advnorm_off_seed0.pt",
    )

    parser.add_argument(
        "--role-b-bc-greedy-weights",
        type=str,
        default="weights/role_b/bc_greedy_seed0.pt",
    )

    # Role C weights
    parser.add_argument(
        "--role-c-weights",
        type=str,
        default="weights/meta_dynaq_v2_seed0.pkl.gz",
    )

    # Offline weights
    parser.add_argument(
        "--offline-bc-weights",
        type=str,
        default="weights/offline_rl_alpha005/bc_policy.pt",
    )

    parser.add_argument(
        "--offline-dqn-weights",
        type=str,
        default="weights/offline_rl_alpha005/offline_dqn.pt",
    )

    parser.add_argument(
        "--offline-iql-weights",
        type=str,
        default="weights/offline_rl_iql/iql_lite.pt",
    )

    # Multi-agent weights
    parser.add_argument(
        "--multi-agent-weights",
        type=str,
        default="weights/multi_agent/idqn_bc_seed0.pt",
    )

    parser.add_argument(
        "--output-json",
        type=str,
        default="logs/run_all_results.json",
    )

    parser.add_argument(
        "--output-csv",
        type=str,
        default="logs/run_all_results.csv",
    )

    parser.add_argument("--skip-role-a", action="store_true")
    parser.add_argument("--skip-role-b", action="store_true")
    parser.add_argument("--skip-role-c", action="store_true")
    parser.add_argument("--skip-offline", action="store_true")
    parser.add_argument("--skip-multi-agent", action="store_true")

    args = parser.parse_args()

    cfg_path = ROOT / args.config
    cfg = Config.from_yaml(str(cfg_path))
    seeds = parse_seeds(args.seeds)

    print("RUN_ALL - IE306 Drone Delivery RL Project")
    print(f"Config: {cfg_path}")
    print(f"Seeds : {seeds}")

    results: Dict[str, Optional[Dict[str, Any]]] = {}

    # Baselines
    maybe_add_baselines(results, cfg, seeds)

    # Role A
    if not args.skip_role_a:
        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role A - DQN",
            weights_path=ROOT / args.role_a_dqn_weights,
        )

        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role A - Double DQN",
            weights_path=ROOT / args.role_a_double_weights,
        )

        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role A - Dueling DQN",
            weights_path=ROOT / args.role_a_dueling_weights,
            prefer_dueling=True,
        )

    # Role B
    if not args.skip_role_b:
        maybe_add_role_b(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role B - A2C",
            weights_path=ROOT / args.role_b_a2c_weights,
        )

        maybe_add_role_b(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role B - A2C BC Fine-tune",
            weights_path=ROOT / args.role_b_bc_finetune_weights,
        )

        maybe_add_role_b(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role B - A2C Advantage Norm Off",
            weights_path=ROOT / args.role_b_advnorm_off_weights,
        )

        maybe_add_role_b(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Role B - BC Greedy",
            weights_path=ROOT / args.role_b_bc_greedy_weights,
        )

    # Role C
    if not args.skip_role_c:
        maybe_add_role_c(
            results=results,
            cfg=cfg,
            seeds=seeds,
            weights_path=ROOT / args.role_c_weights,
        )

    # Offline RL
    if not args.skip_offline:
        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Offline RL - Behavior Cloning",
            weights_path=ROOT / args.offline_bc_weights,
        )

        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Offline RL - Naive DQN",
            weights_path=ROOT / args.offline_dqn_weights,
        )

        maybe_add_generic_torch_policy(
            results=results,
            cfg=cfg,
            seeds=seeds,
            name="Offline RL - IQL-lite",
            weights_path=ROOT / args.offline_iql_weights,
        )

    # Multi-agent
    if not args.skip_multi_agent:
        maybe_add_multi_agent(
            results=results,
            cfg=cfg,
            seeds=seeds,
            weights_path=ROOT / args.multi_agent_weights,
        )

    print_comparison_table(results)

    save_outputs(
        results=results,
        output_json=ROOT / args.output_json,
        output_csv=ROOT / args.output_csv,
    )


if __name__ == "__main__":
    main()