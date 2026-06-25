from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    import drone_dispatch_env
    from drone_dispatch_env import Config, DroneDispatchEnv
    from drone_dispatch_env.evaluate import evaluate
    from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. "
        "Run `pip install -e .` from the repository root."
    ) from e

from policy_interface import TrainedPriorityDynaQPolicy


def parse_seeds(seed_text: str):
    return [
        int(s.strip())
        for s in seed_text.split(",")
        if s.strip() != ""
    ]


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)

    if path.is_absolute():
        return path

    return ROOT / path


def get_state_and_action_size(cfg: Config):
    env = DroneDispatchEnv(cfg)

    obs, info = env.reset(seed=0)

    state_size = (
        len(obs["drones"].flatten())
        + len(obs["orders"].flatten())
        + len(obs["grid"].flatten())
        + len(obs["time"].flatten())
    )

    action_size = len(obs["action_mask"])

    return state_size, action_size


def evaluate_trained_policy(
    weights_path: Path,
    cfg: Config,
    seeds,
):
    state_size, action_size = get_state_and_action_size(cfg)

    policy = TrainedPriorityDynaQPolicy(
        state_size=state_size,
        action_size=action_size,
        weights_path=str(weights_path),
    )

    results = evaluate(
        policy,
        cfg,
        seeds=seeds,
    )

    return results


def evaluate_baselines(
    cfg: Config,
    seeds,
):
    baselines = {
        "random": RandomPolicy(cfg, seed=0),
        "greedy_nearest": GreedyNearest(cfg),
        "milp_rolling": MILPRolling(cfg),
    }

    results = {}

    for name, policy in baselines.items():
        print(f"\n[baseline] Evaluating {name}")

        results[name] = evaluate(
            policy,
            cfg,
            seeds=seeds,
        )

        print(
            json.dumps(
                results[name]["mean"],
                indent=2,
            )
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="weights/priority_dynaq_seed0.pt",
        help="Path to trained Priority Dyna-Q model.",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/eval_standard.yaml",
        help="Path to evaluation config.",
    )

    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated evaluation seeds.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="logs/priority_dynaq_eval_seed0.json",
        help="Path to save evaluation results.",
    )

    parser.add_argument(
        "--with-baselines",
        action="store_true",
        help="Also evaluate random, greedy_nearest, and milp_rolling.",
    )

    args = parser.parse_args()

    config_path = resolve_path(args.config)
    weights_path = resolve_path(args.weights)
    output_path = resolve_path(args.output)

    seeds = parse_seeds(args.seeds)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Evaluation config not found: {config_path}"
        )

    if not weights_path.exists():
        raise FileNotFoundError(
            f"Trained weights not found: {weights_path}"
        )

    cfg = Config.from_yaml(str(config_path))

    print("\nEvaluating Role C Priority Dyna-Q")
    print(f"Weights: {weights_path}")
    print(f"Config : {config_path}")
    print(f"Seeds  : {seeds}")

    priority_results = evaluate_trained_policy(
        weights_path=weights_path,
        cfg=cfg,
        seeds=seeds,
    )

    all_results = {
        "priority_dynaq": priority_results,
    }

    print("\nPriority Dyna-Q mean metrics:")
    print(
        json.dumps(
            priority_results["mean"],
            indent=2,
        )
    )

    if args.with_baselines:
        baseline_results = evaluate_baselines(
            cfg=cfg,
            seeds=seeds,
        )

        all_results.update(baseline_results)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(f"\nEvaluation saved to: {output_path}")


if __name__ == "__main__":
    main()