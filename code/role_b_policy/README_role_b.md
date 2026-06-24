# Role B — Policy-Based RL and Actor-Critic

This folder contains the Role B implementation for the IE306 Drone Delivery term project.

Role B is responsible for policy-based and actor-critic methods:

- REINFORCE on DroneDispatch-v0
- A2C with GAE / advantage normalization on DroneDispatch-v0
- DDPG on DroneControl-v0

Main evaluation metric:
- cost_per_order

Baselines for comparison:
- random
- greedy_nearest
- milp_rolling

Planned ablation:
- advantage normalization on/off