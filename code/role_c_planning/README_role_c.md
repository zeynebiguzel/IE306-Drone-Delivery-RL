# Role C — Planning / Model-Based Acceleration

## Method Summary

This role implements the **Planning / Model-Based Acceleration** component for the IE306 Drone Delivery RL project.

The final submitted policy is a **Priority Dyna-Q inspired rollout planner**. The implementation contains a tabular Priority Dyna-Q training module with:

* discretized state encoder,
* tabular Q-function,
* deterministic model memory,
* predecessor tracking,
* prioritized sweeping planning queue,
* configurable planning steps.

During evaluation, the final policy uses a safer planning wrapper:

1. It starts from the official `greedy_nearest` dispatch action.
2. It checks whether the selected drone is risky for the selected order using a one-step battery-risk model.
3. If the selected drone is risky, the planner tries to keep the same order but assign it to another valid drone with safer battery level.
4. If no safer drone exists, it may send the risky drone to charge.
5. Otherwise, it keeps the original greedy action.

This is a planning/model-based policy rather than a DQN policy. The main improvement comes from a conservative rollout-style battery guard that reduces depletion risk while preserving the strong assignment behavior of the greedy baseline.

## Files

Important Role C files:

```text
code/role_c_planning/network.py
code/role_c_planning/priority_dynaq.py
code/role_c_planning/policy_interface.py
code/role_c_planning/train_priority_dynaq.py
code/role_c_planning/evaluate_priority_dynaq.py
code/role_c_planning/plot_priority_dynaq_results.py
code/role_c_planning/make_role_c_final_table.py
configs/priority_dynaq.yaml
weights/priority_dynaq_seed0.pkl.gz
weights/priority_dynaq_seed1.pkl.gz
weights/priority_dynaq_seed2.pkl.gz
logs/priority_dynaq_seed0.csv
logs/priority_dynaq_seed1.csv
logs/priority_dynaq_seed2.csv
logs/priority_dynaq_eval_seed0.json
logs/priority_dynaq_eval_seed1.json
logs/priority_dynaq_eval_seed2.json
logs/priority_dynaq_learning_curve_return.png
logs/priority_dynaq_learning_curve_cost.png
logs/priority_dynaq_learning_curve_success.png
logs/priority_dynaq_model_size.png
logs/role_c_final_baseline_table.csv
```

## Training Commands

The model was trained over three random seeds.

```bash
python code/role_c_planning/train_priority_dynaq.py --config configs/priority_dynaq.yaml --seed 0
python code/role_c_planning/train_priority_dynaq.py --config configs/priority_dynaq.yaml --seed 1
python code/role_c_planning/train_priority_dynaq.py --config configs/priority_dynaq.yaml --seed 2
```

These commands generate:

```text
weights/priority_dynaq_seed0.pkl.gz
weights/priority_dynaq_seed1.pkl.gz
weights/priority_dynaq_seed2.pkl.gz
logs/priority_dynaq_seed0.csv
logs/priority_dynaq_seed1.csv
logs/priority_dynaq_seed2.csv
```

## Evaluation Commands

The trained Role C policy and baselines can be evaluated with:

```bash
python code/role_c_planning/evaluate_priority_dynaq.py --weights weights/priority_dynaq_seed0.pkl.gz --config configs/eval_standard.yaml --seeds 0,1,2 --output logs/priority_dynaq_eval_seed0.json --with-baselines

python code/role_c_planning/evaluate_priority_dynaq.py --weights weights/priority_dynaq_seed1.pkl.gz --config configs/eval_standard.yaml --seeds 0,1,2 --output logs/priority_dynaq_eval_seed1.json --with-baselines

python code/role_c_planning/evaluate_priority_dynaq.py --weights weights/priority_dynaq_seed2.pkl.gz --config configs/eval_standard.yaml --seeds 0,1,2 --output logs/priority_dynaq_eval_seed2.json --with-baselines
```

## Plot and Table Commands

Learning curves and final result tables are generated with:

```bash
python code/role_c_planning/plot_priority_dynaq_results.py
python code/role_c_planning/make_role_c_final_table.py
```

Generated outputs:

```text
logs/priority_dynaq_learning_curve_return.png
logs/priority_dynaq_learning_curve_cost.png
logs/priority_dynaq_learning_curve_success.png
logs/priority_dynaq_model_size.png
logs/priority_dynaq_eval_summary.csv
logs/role_c_final_baseline_table.csv
```

## Final Baseline Comparison

Evaluation was performed on `configs/eval_standard.yaml` over seeds `0,1,2`.

| Policy         | Cost / Order | Success | On-time | Delivered | Dropped | Depletion |  Return |
| -------------- | -----------: | ------: | ------: | --------: | ------: | --------: | ------: |
| priority_dynaq |       4.4373 |   0.847 |   0.903 |    119.33 |   21.33 |      3.33 | 1211.55 |
| random         |      18.7804 |   0.653 |   0.890 |     39.67 |   21.67 |      8.00 | -168.33 |
| greedy_nearest |       4.5700 |   0.855 |   0.903 |    118.33 |   20.00 |      4.00 | 1183.26 |
| milp_rolling   |       4.7223 |   0.836 |   0.911 |    118.00 |   23.00 |      3.33 | 1173.00 |

The final Role C planner beats the `greedy_nearest` baseline on the main project metric:

```text
priority_dynaq cost_per_order = 4.4373
greedy_nearest cost_per_order = 4.5700
```

It also improves episode return:

```text
priority_dynaq episode_return = 1211.55
greedy_nearest episode_return = 1183.26
```

The improvement mainly comes from reducing battery depletion events:

```text
priority_dynaq depletion_events = 3.33
greedy_nearest depletion_events = 4.00
```

## Ablation

For Role C, the ablation is a rollout-depth comparison:

| Variant         | Interpretation                                      | Cost / Order | Depletion |  Return |
| --------------- | --------------------------------------------------- | -----------: | --------: | ------: |
| rollout depth 0 | greedy_nearest only, no planner correction          |       4.5700 |      4.00 | 1183.26 |
| rollout depth 1 | greedy action + one-step battery/reassignment guard |       4.4373 |      3.33 | 1211.55 |

The depth-1 rollout improves cost per delivered order by checking the greedy assignment with a simple one-step battery-risk model. The planner only intervenes when the selected drone appears risky and another valid drone can serve the same order more safely.

## What Broke and How It Was Diagnosed

The first pure tabular Priority Dyna-Q policy performed poorly because the full drone dispatch state is large and highly continuous. Even after discretization, the table-based agent had weak generalization and delivered far fewer orders than the greedy baseline.

The diagnosis was based on evaluation metrics:

* low number of delivered orders,
* high cost per order,
* poor episode return.

To fix this, the final policy uses a strong greedy warm-start and applies planning only as a conservative correction. This keeps the strong dispatch behavior of `greedy_nearest` while reducing battery risk through a one-step rollout-style guard.

## Method Origin

The implementation is based on Sutton’s Dyna-Q and prioritized sweeping ideas from model-based reinforcement learning. Dyna-style methods combine direct learning from real transitions with planning updates from a learned model. Prioritized sweeping focuses planning updates on state-action pairs with large temporal-difference errors.

In this implementation, the training module includes model memory and prioritized planning, while the final evaluation policy uses a lightweight rollout planner because it was more robust in the large drone dispatch environment.
