from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ROWS = [
    {
        "policy": "priority_dynaq",
        "cost_per_order": 4.437281117892117,
        "success_rate": 0.8473529369400513,
        "ontime_rate": 0.9032724229254541,
        "mean_delivery_time": 34.34507204214649,
        "energy_per_order": 0.21651153421202787,
        "depletion_events": 3.3333333333333335,
        "idle_pct": 0.060579481259798264,
        "charger_utilization": 0.1297871196015461,
        "n_delivered": 119.33333333333333,
        "n_dropped": 21.333333333333332,
        "episode_return": 1211.5516666666672,
    },
    {
        "policy": "random",
        "cost_per_order": 18.78035406746031,
        "success_rate": 0.6527568062228323,
        "ontime_rate": 0.890079365079365,
        "mean_delivery_time": 39.175165343915346,
        "energy_per_order": 0.3147555224867669,
        "depletion_events": 8.0,
        "idle_pct": 0.06802855804182396,
        "charger_utilization": 0.06690761693901064,
        "n_delivered": 39.666666666666664,
        "n_dropped": 21.666666666666668,
        "episode_return": -168.326,
    },
    {
        "policy": "greedy_nearest",
        "cost_per_order": 4.570042801915485,
        "success_rate": 0.8549158221641059,
        "ontime_rate": 0.902764421909452,
        "mean_delivery_time": 34.086563025128456,
        "energy_per_order": 0.2182026060941716,
        "depletion_events": 4.0,
        "idle_pct": 0.06667568662830013,
        "charger_utilization": 0.12737547359538326,
        "n_delivered": 118.33333333333333,
        "n_dropped": 20.0,
        "episode_return": 1183.2566666666671,
    },
    {
        "policy": "milp_rolling",
        "cost_per_order": 4.722329311461045,
        "success_rate": 0.8363957881199261,
        "ontime_rate": 0.9109105399237051,
        "mean_delivery_time": 32.511990838055674,
        "energy_per_order": 0.21530528389326242,
        "depletion_events": 3.3333333333333335,
        "idle_pct": 0.06556271150824385,
        "charger_utilization": 0.1325432733784415,
        "n_delivered": 118.0,
        "n_dropped": 23.0,
        "episode_return": 1172.9980000000005,
    },
]


def save_csv():
    output_path = ROOT / "logs" / "role_c_final_baseline_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(ROWS[0].keys()),
        )

        writer.writeheader()
        writer.writerows(ROWS)

    return output_path


def print_table():
    print("\nROLE C FINAL BASELINE COMPARISON")
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

    for row in ROWS:
        print(
            f"{row['policy']:<18} | "
            f"{row['cost_per_order']:>10.4f} | "
            f"{row['success_rate']:>8.3f} | "
            f"{row['ontime_rate']:>8.3f} | "
            f"{row['n_delivered']:>10.2f} | "
            f"{row['n_dropped']:>8.2f} | "
            f"{row['depletion_events']:>10.2f} | "
            f"{row['episode_return']:>10.2f}"
        )

    print("-" * 112)


def main():
    print_table()
    output_path = save_csv()
    print(f"\nSaved final table: {output_path}")


if __name__ == "__main__":
    main()