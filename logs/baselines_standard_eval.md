# Baseline Evaluation on Standard Config

This file records the baseline evaluation results for the IE306 Drone Delivery RL project.

Evaluation command format:

```bash
python -m drone_dispatch_env.run_eval --config configs/eval_standard.yaml --seeds 0,1,2 --policy <policy_name>
```

Evaluation seeds:

```text
0, 1, 2
```

## Test Verification

Before running the baseline policies, the simulator was tested from the repository root:

```bash
python -m pytest tests -q
```

Result:

```text
14 passed
```

This confirms that the simulator environment works correctly inside the shared GitHub repository.

## Baseline Results

| Policy         | cost_per_order | success_rate | ontime_rate | mean_delivery_time | energy_per_order | depletion_events | n_delivered | n_dropped | episode_return |
| -------------- | -------------: | -----------: | ----------: | -----------------: | ---------------: | ---------------: | ----------: | --------: | -------------: |
| random         |      18.780354 |     0.652757 |    0.890079 |          39.175165 |         0.314756 |         8.000000 |   39.666667 | 21.666667 |    -168.326000 |
| greedy_nearest |       4.570043 |     0.854916 |    0.902764 |          34.086563 |         0.218203 |         4.000000 |  118.333333 | 20.000000 |    1183.256667 |
| milp_rolling   |       4.722329 |     0.836396 |    0.910911 |          32.511991 |         0.215305 |         3.333333 |  118.000000 | 23.000000 |    1172.998000 |

## Short Interpretation

The random policy performs poorly because it does not make informed dispatching decisions. Its cost_per_order is much higher and it delivers far fewer orders.

The greedy_nearest policy is the strongest baseline according to the main metric, cost_per_order. It achieves a cost_per_order of 4.570043 and delivers 118.33 orders on average.

The milp_rolling policy has slightly better mean delivery time, energy_per_order, and depletion_events than greedy_nearest. However, its cost_per_order is higher than greedy_nearest in this evaluation. Since cost_per_order is the primary metric, greedy_nearest is used as the main baseline to beat for the learned Role B methods.
