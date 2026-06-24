import numpy as np


class PriorityReplayBuffer:
    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, next_state, done, priority):
        self.memory.append(
            (state, action, reward, next_state, done, priority)
        )

    def sample(self, n):
        if len(self.memory) == 0:
            return []

        priorities = np.array([m[5] for m in self.memory])

        priorities = priorities + 1e-6
        probabilities = priorities / priorities.sum()

        idx = np.random.choice(
            len(self.memory),
            size=min(n, len(self.memory)),
            replace=False,
            p=probabilities
        )

        return [self.memory[i] for i in idx]


class PriorityDynaQAgent:
    def __init__(self):
        self.model_memory = {}
        self.replay_buffer = PriorityReplayBuffer()

    def store_transition(
        self,
        state,
        action,
        reward,
        next_state,
        done,
        priority
    ):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done,
            priority
        )

        key = (tuple(state), action)

        self.model_memory[key] = (
            reward,
            next_state,
            done
        )

    def planning_sample(self, n):
        return self.replay_buffer.sample(n)