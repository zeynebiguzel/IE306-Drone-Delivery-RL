import subprocess
import sys

seeds = [0, 1, 2]

algorithms = [
    "train_dqn.py",
    "train_double_dqn.py",
    "train_dueling_dqn.py"
]

for algorithm in algorithms:

    print("\n" + "=" * 50)
    print(f"RUNNING: {algorithm}")
    print("=" * 50)

    for seed in seeds:

        print(f"\nSeed {seed}")

        subprocess.run(
            [
                sys.executable,
                f"code/role_a_dqn/{algorithm}",
                "--seed",
                str(seed)
            ],
            check=True
        )

print("\nALL EXPERIMENTS FINISHED")