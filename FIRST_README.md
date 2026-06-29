# IE306 Drone Delivery RL Project

This repository contains the IE306 Reinforcement Learning and Dynamic Optimization term project for city-scale drone delivery dispatching.

## Installation

```bash
pip install -r requirements.txt
Reproduce Results

Run the full evaluation table with:

python run_all.py --config configs/eval_standard.yaml --seeds 0,1,2

The script evaluates baseline policies, Role A methods, Role B methods, Role C planning method, Offline RL models, and the Multi-Agent model.

Results are saved to:

logs/run_all_results.csv
logs/run_all_results.json
Main Files
code/: source code for all methods
configs/: experiment configuration files
weights/: trained model files
logs/: training and evaluation logs
datasets/: offline RL dataset
run_all.py: reproducibility script
REPORT.md: project report
REPORT.docx: project report in Word format
ROLES.md: team role ownership
AI_USE.md: AI tool usage declaration