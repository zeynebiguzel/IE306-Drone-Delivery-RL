import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_algorithm(prefix):

    seed0 = pd.read_csv(
        f"logs/{prefix}_seed0.csv"
    )

    seed1 = pd.read_csv(
        f"logs/{prefix}_seed1.csv"
    )

    seed2 = pd.read_csv(
        f"logs/{prefix}_seed2.csv"
    )

    rewards = np.vstack([
        seed0["reward"].values,
        seed1["reward"].values,
        seed2["reward"].values
    ])

    mean_reward = rewards.mean(axis=0)
    std_reward = rewards.std(axis=0)
    return mean_reward, std_reward

# Load Results
dqn_mean, dqn_std = load_algorithm(
    "dqn"
)

double_mean, double_std = load_algorithm(
    "double_double_dqn"
)

dueling_mean, dueling_std = load_algorithm(
    "dueling_dqn"
)

episodes = np.arange(
    1,
    len(dqn_mean) + 1
)

# Plot
plt.figure(figsize=(10, 6))

# DQN
plt.plot(
    episodes,
    dqn_mean,
    label="DQN"
)

plt.fill_between(
    episodes,
    dqn_mean - dqn_std,
    dqn_mean + dqn_std,
    alpha=0.2
)

# Double DQN
plt.plot(
    episodes,
    double_mean,
    label="Double DQN"
)

plt.fill_between(
    episodes,
    double_mean - double_std,
    double_mean + double_std,
    alpha=0.2
)

# Dueling DQN
plt.plot(
    episodes,
    dueling_mean,
    label="Dueling DQN"
)

plt.fill_between(
    episodes,
    dueling_mean - dueling_std,
    dueling_mean + dueling_std,
    alpha=0.2
)

plt.xlabel("Episode")
plt.ylabel("Reward")

plt.title(
    "Learning Curves (Mean ± Std over 3 Seeds)"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "learning_curves.png"
)

plt.show()

# Final Table
print("\nFINAL REWARD SUMMARY\n")

print(
    f"DQN          : "
    f"{dqn_mean[-1]:.2f} ± {dqn_std[-1]:.2f}"
)

print(
    f"Double DQN   : "
    f"{double_mean[-1]:.2f} ± {double_std[-1]:.2f}"
)

print(
    f"Dueling DQN  : "
    f"{dueling_mean[-1]:.2f} ± {dueling_std[-1]:.2f}"
)