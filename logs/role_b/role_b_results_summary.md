# Role B Results Summary

This file summarizes the Role B experiments for the IE306 Drone Delivery RL project.

## Evaluation Setup

All policies were evaluated using the standard evaluation configuration:

```bash id="rsy7ub"
python -m drone_dispatch_env.run_eval --config configs/eval_standard.yaml --seeds 0,1,2 --policy <policy_name>
```

For learned Role B models, the following evaluation script was used:

```bash id="iqai6c"
python code/role_b_policy/eval_a2c.py --weights <weights_path> --config configs/eval_standard.yaml --seeds 0,1,2 --output <output_path>
```

Evaluation seeds:

```text id="ic92oq"
0, 1, 2
```

The primary metric is:

```text id="r2x2gb"
cost_per_order
```

Lower cost_per_order is better.

## Main Results

| Method                              | cost_per_order | success_rate | ontime_rate | n_delivered | n_dropped | episode_return |
| ----------------------------------- | -------------: | -----------: | ----------: | ----------: | --------: | -------------: |
| random                              |         18.780 |        0.653 |       0.890 |       39.67 |     21.67 |        -168.33 |
| greedy_nearest                      |          4.570 |        0.855 |       0.903 |      118.33 |     20.00 |        1183.26 |
| milp_rolling                        |          4.722 |        0.836 |       0.911 |      118.00 |     23.00 |        1173.00 |
| vanilla A2C                         |         24.371 |        0.618 |       0.948 |       27.67 |     17.67 |        -267.35 |
| behavior cloning from greedy        |         30.031 |        0.453 |       0.937 |       62.67 |     73.33 |        -446.05 |
| BC warm-start + A2C fine-tuning     |         26.713 |        0.409 |       0.807 |       55.33 |     80.00 |        -700.71 |
| A2C without advantage normalization |         35.832 |        0.438 |       0.948 |       25.00 |     32.33 |        -527.42 |

## Interpretation

The greedy_nearest baseline was the strongest policy according to the primary metric, cost_per_order. It achieved a cost_per_order of 4.570, while milp_rolling achieved 4.722. Therefore, greedy_nearest is the main baseline to beat.

The vanilla A2C pipeline worked correctly. The model interacted with the environment, used the action mask, saved weights, and produced evaluation metrics. However, after 50,000 training steps, vanilla A2C obtained a cost_per_order of 24.371, which was worse than both random and greedy_nearest. This shows that learning a competitive dispatching policy from scratch is difficult in this environment.

One reason is that DroneDispatch-v0 has a large discrete action space, masked invalid actions, delayed rewards, battery constraints, and order deadlines. The agent must learn all of these interactions at the same time. In contrast, greedy_nearest already provides a strong short-term dispatching rule.

Behavior cloning was tested as a warm-start strategy. First, the actor network was trained to imitate the greedy_nearest policy using supervised learning. The imitation accuracy reached 0.337 after 8 epochs. However, this was not enough to reproduce greedy_nearest reliably. The behavior cloning policy achieved a cost_per_order of 30.031 and produced many dropped orders.

Then, the behavior cloning model was used as an initial model for A2C fine-tuning. This improved over the direct behavior cloning policy, reducing cost_per_order from 30.031 to 26.713. However, it still did not become competitive with vanilla A2C, random, or greedy_nearest. This suggests that low imitation accuracy and small action errors can cause cascading dispatching failures.

The advantage normalization ablation showed that removing advantage normalization made A2C worse. With advantage normalization, vanilla A2C achieved a cost_per_order of 24.371. Without advantage normalization, cost_per_order increased to 35.832. Therefore, advantage normalization was useful as a stabilization technique, even though it was not sufficient to solve the full dispatching problem.

Overall, Role B successfully implemented and evaluated an actor-critic pipeline, behavior cloning pretraining, BC warm-start fine-tuning, and an advantage normalization ablation. The learned policies did not beat the greedy_nearest baseline, but the experiments provide a clear diagnosis of why policy-gradient learning from scratch is challenging in this dispatching environment.
