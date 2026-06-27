from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parent))


try:
    import drone_dispatch_env
    from drone_dispatch_env import Config
    from drone_dispatch_env.evaluate import evaluate
    from drone_dispatch_env.baselines import RandomPolicy, GreedyNearest, MILPRolling
except ImportError as e:
    raise ImportError(
        "Could not import drone_dispatch_env. "
        "Run `pip install -e .` from the repository root."
    ) from e


from meta_dynaq_planner import TrainedMetaDynaQPlanner


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


def evaluate_meta_policy(
    weights_path: Path,
    cfg: Config,
    seeds,
):
    policy = TrainedMetaDynaQPlanner(
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


def print_compact_table(all_results):
    print("\nROLE C META DYNA-Q BASELINE COMPARISON")
    print("-" * 112)

    print(
        f"{'Policy':<18} | "
        f"{'Cost/Order':>10} | "
        f"{'Success':>8} | "
        f"{'On-time':>8} | "
        f"{'Delivered':>10} | "
        f"{'Dropped':>8} | "
        f"{'Depletion':>10} | "
        f"{'Return':>10}"
    )

    print("-" * 112)

    preferred = [
        "meta_dynaq",
        "random",
        "greedy_nearest",
        "milp_rolling",
    ]

    for name in preferred:
        if name not in all_results:
            continue

        mean = all_results[name]["mean"]

        print(
            f"{name:<18} | "
            f"{float(mean.get('cost_per_order', 0.0)):>10.4f} | "
            f"{float(mean.get('success_rate', 0.0)):>8.3f} | "
            f"{float(mean.get('ontime_rate', 0.0)):>8.3f} | "
            f"{float(mean.get('n_delivered', 0.0)):>10.2f} | "
            f"{float(mean.get('n_dropped', 0.0)):>8.2f} | "
            f"{float(mean.get('depletion_events', 0.0)):>10.2f} | "
            f"{float(mean.get('episode_return', 0.0)):>10.2f}"
        )

    print("-" * 112)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        type=str,
        default="weights/meta_dynaq_seed0.pkl.gz",
        help="Path to trained Meta Dyna-Q model.",
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
        default="logs/meta_dynaq_eval_seed0.json",
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

    print("\nEvaluating Role C Meta Dyna-Q Planner")
    print(f"Weights: {weights_path}")
    print(f"Config : {config_path}")
    print(f"Seeds  : {seeds}")

    meta_results = evaluate_meta_policy(
        weights_path=weights_path,
        cfg=cfg,
        seeds=seeds,
    )

    all_results = {
        "meta_dynaq": meta_results,
    }

    print("\nMeta Dyna-Q mean metrics:")
    print(
        json.dumps(
            meta_results["mean"],
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

    print_compact_table(all_results)

    print(f"\nEvaluation saved to: {output_path}")


if __name__ == "__main__":
    main()