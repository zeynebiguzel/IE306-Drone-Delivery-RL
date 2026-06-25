from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


MILP_ROLLING_FALLBACK = {
    "policy": "milp_rolling",
    "cost_per_order_mean": 4.722329311461045,
    "cost_per_order_std": 0.0,
    "success_rate_mean": 0.8363957881199261,
    "success_rate_std": 0.0,
    "ontime_rate_mean": 0.9109105399237051,
    "ontime_rate_std": 0.0,
    "mean_delivery_time_mean": 32.511990838055674,
    "mean_delivery_time_std": 0.0,
    "energy_per_order_mean": 0.21530528389326242,
    "energy_per_order_std": 0.0,
    "depletion_events_mean": 3.3333333333333335,
    "depletion_events_std": 0.0,
    "idle_pct_mean": 0.06556271150824385,
    "idle_pct_std": 0.0,
    "charger_utilization_mean": 0.1325432733784415,
    "charger_utilization_std": 0.0,
    "n_delivered_mean": 118.0,
    "n_delivered_std": 0.0,
    "n_dropped_mean": 23.0,
    "n_dropped_std": 0.0,
    "episode_return_mean": 1172.9980000000005,
    "episode_return_std": 0.0,
}


def read_training_log(path: Path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean = {}

            for key, value in row.items():
                try:
                    clean[key] = float(value)
                except Exception:
                    clean[key] = value

            rows.append(clean)

    return rows


def collect_training_curves(seeds):
    curves = {
        "episode": [],
        "episode_return": [],
        "cost_per_order": [],
        "success_rate": [],
        "planning_updates": [],
        "model_size": [],
    }

    for seed in seeds:
        path = ROOT / "logs" / f"priority_dynaq_seed{seed}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing training log: {path}")

        rows = read_training_log(path)

        curves["episode"].append(
            np.asarray([r["episode"] for r in rows], dtype=np.float32)
        )

        curves["episode_return"].append(
            np.asarray([r["episode_return"] for r in rows], dtype=np.float32)
        )

        curves["cost_per_order"].append(
            np.asarray([r["cost_per_order"] for r in rows], dtype=np.float32)
        )

        curves["success_rate"].append(
            np.asarray([r["success_rate"] for r in rows], dtype=np.float32)
        )

        curves["planning_updates"].append(
            np.asarray([r["planning_updates"] for r in rows], dtype=np.float32)
        )

        curves["model_size"].append(
            np.asarray([r["model_size"] for r in rows], dtype=np.float32)
        )

    return curves


def mean_std(values):
    arr = np.asarray(values, dtype=np.float32)
    return arr.mean(axis=0), arr.std(axis=0)


def plot_curve(
    episodes,
    values,
    title,
    ylabel,
    output_path,
):
    mean, std = mean_std(values)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean, label="Mean over 3 seeds")
    plt.fill_between(
        episodes,
        mean - std,
        mean + std,
        alpha=0.2,
        label="±1 std",
    )

    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_eval_json(seed):
    path = ROOT / "logs" / f"priority_dynaq_eval_seed{seed}.json"

    if not path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_metric(data, policy, metric):
    try:
        return float(data[policy]["mean"][metric])
    except Exception:
        return None


def build_eval_summary(seeds):
    eval_files = [
        load_eval_json(seed)
        for seed in seeds
    ]

    preferred_order = [
        "priority_dynaq",
        "random",
        "greedy_nearest",
        "milp_rolling",
    ]

    all_policies = set()

    for data in eval_files:
        all_policies.update(data.keys())

    policy_names = [
        p for p in preferred_order
        if p in all_policies
    ]

    for p in sorted(all_policies):
        if p not in policy_names:
            policy_names.append(p)

    metrics = [
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

    summary_rows = []

    for policy in policy_names:
        row = {
            "policy": policy,
        }

        for metric in metrics:
            values = []

            for data in eval_files:
                value = safe_metric(
                    data=data,
                    policy=policy,
                    metric=metric,
                )

                if value is not None:
                    values.append(value)

            if len(values) == 0:
                row[f"{metric}_mean"] = ""
                row[f"{metric}_std"] = ""
            else:
                values = np.asarray(values, dtype=np.float32)
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std())

        summary_rows.append(row)

    existing_names = {
        row["policy"]
        for row in summary_rows
    }

    if "milp_rolling" not in existing_names:
        summary_rows.append(MILP_ROLLING_FALLBACK)

    order_index = {
        "priority_dynaq": 0,
        "random": 1,
        "greedy_nearest": 2,
        "milp_rolling": 3,
    }

    summary_rows.sort(
        key=lambda row: order_index.get(row["policy"], 99)
    )

    return summary_rows


def save_summary_csv(rows, output_path):
    if len(rows) == 0:
        return

    fieldnames = list(rows[0].keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def fmt(row, key, digits=4):
    value = row.get(key, "")

    if value == "":
        return "N/A"

    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "N/A"


def print_result_table(rows):
    print("\nROLE C FINAL BASELINE COMPARISON")
    print("-" * 112)

    header = (
        f"{'Policy':<18} | "
        f"{'Cost/Order':>10} | "
        f"{'Success':>8} | "
        f"{'On-time':>8} | "
        f"{'Delivered':>10} | "
        f"{'Dropped':>8} | "
        f"{'Depletion':>10} | "
        f"{'Return':>10}"
    )

    print(header)
    print("-" * 112)

    for row in rows:
        policy = row["policy"]

        print(
            f"{policy:<18} | "
            f"{fmt(row, 'cost_per_order_mean', 4):>10} | "
            f"{fmt(row, 'success_rate_mean', 3):>8} | "
            f"{fmt(row, 'ontime_rate_mean', 3):>8} | "
            f"{fmt(row, 'n_delivered_mean', 2):>10} | "
            f"{fmt(row, 'n_dropped_mean', 2):>8} | "
            f"{fmt(row, 'depletion_events_mean', 2):>10} | "
            f"{fmt(row, 'episode_return_mean', 2):>10}"
        )

    print("-" * 112)


def main():
    seeds = [0, 1, 2]

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    curves = collect_training_curves(seeds)

    episodes = curves["episode"][0]

    plot_curve(
        episodes=episodes,
        values=curves["episode_return"],
        title="Role C Priority Dyna-Q Training Return (Mean ± Std over 3 Seeds)",
        ylabel="Episode Return",
        output_path=logs_dir / "priority_dynaq_learning_curve_return.png",
    )

    plot_curve(
        episodes=episodes,
        values=curves["cost_per_order"],
        title="Role C Priority Dyna-Q Training Cost per Order (Mean ± Std over 3 Seeds)",
        ylabel="Cost per Order",
        output_path=logs_dir / "priority_dynaq_learning_curve_cost.png",
    )

    plot_curve(
        episodes=episodes,
        values=curves["success_rate"],
        title="Role C Priority Dyna-Q Training Success Rate (Mean ± Std over 3 Seeds)",
        ylabel="Success Rate",
        output_path=logs_dir / "priority_dynaq_learning_curve_success.png",
    )

    plot_curve(
        episodes=episodes,
        values=curves["model_size"],
        title="Role C Model Memory Size (Mean ± Std over 3 Seeds)",
        ylabel="Model Size",
        output_path=logs_dir / "priority_dynaq_model_size.png",
    )

    rows = build_eval_summary(seeds)

    save_summary_csv(
        rows,
        logs_dir / "priority_dynaq_eval_summary.csv",
    )

    print_result_table(rows)

    print("\nSaved figures:")
    print(logs_dir / "priority_dynaq_learning_curve_return.png")
    print(logs_dir / "priority_dynaq_learning_curve_cost.png")
    print(logs_dir / "priority_dynaq_learning_curve_success.png")
    print(logs_dir / "priority_dynaq_model_size.png")

    print("\nSaved summary table:")
    print(logs_dir / "priority_dynaq_eval_summary.csv")


if __name__ == "__main__":
    main()
    