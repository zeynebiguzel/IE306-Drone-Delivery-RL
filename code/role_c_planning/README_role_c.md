This folder contains the Role C implementation for the IE306 Drone Delivery term project.

Role C is responsible for model-based reinforcement learning and planning methods.

Methods:
- Dyna-Q+
- Priority-Based Dyna-Q+

Main evaluation metric:
- cost_per_order

Baselines for comparison:
- random
- greedy_nearest
- milp_rolling
- Dyna-Q+

Planned ablation:
- planning steps (n = 0, 5, 10, 20)
- priority replay on/off
- urgency weighting on/off