from __future__ import annotations

import gzip
import heapq
import os
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from network import StateEncoder, TabularQFunction


Transition = Tuple[
    tuple,
    int,
    float,
    tuple,
    bool,
    np.ndarray,
    np.ndarray,
]


class ModelMemory:
    """
    Deterministic learned model for Dyna-style planning.

    Stores:
        (state_key, action) -> transition

    Also stores predecessor links:
        next_state_key -> {(previous_state_key, previous_action), ...}
    """

    def __init__(self):
        self.transitions: Dict[Tuple[tuple, int], Transition] = {}
        self.predecessors = defaultdict(set)

    def store(
        self,
        state_key: tuple,
        action: int,
        reward: float,
        next_state_key: tuple,
        done: bool,
        action_mask,
        next_action_mask,
    ) -> None:
        action = int(action)

        transition = (
            state_key,
            action,
            float(reward),
            next_state_key,
            bool(done),
            np.asarray(action_mask, dtype=np.float32),
            np.asarray(next_action_mask, dtype=np.float32),
        )

        self.transitions[(state_key, action)] = transition
        self.predecessors[next_state_key].add((state_key, action))

    def get(self, state_key: tuple, action: int) -> Optional[Transition]:
        return self.transitions.get((state_key, int(action)), None)

    def get_predecessors(self, state_key: tuple) -> List[Tuple[tuple, int]]:
        return list(self.predecessors.get(state_key, []))

    def __len__(self) -> int:
        return len(self.transitions)


class PriorityQueue:
    """
    Max-priority queue for prioritized sweeping.
    """

    def __init__(self):
        self.heap = []
        self.counter = 0

    def push(self, priority: float, state_key: tuple, action: int) -> None:
        priority = float(priority)

        if priority <= 0.0:
            return

        self.counter += 1

        heapq.heappush(
            self.heap,
            (-priority, self.counter, state_key, int(action)),
        )

    def pop(self):
        priority, _, state_key, action = heapq.heappop(self.heap)

        return -priority, state_key, action

    def empty(self) -> bool:
        return len(self.heap) == 0

    def __len__(self) -> int:
        return len(self.heap)


class PriorityDynaQAgent:
    """
    Role C agent: Priority Dyna-Q with tabular Q-values.

    Correct Role C logic:
        1. Select action with epsilon-greedy over valid action mask.
        2. Execute real transition.
        3. Direct Q-learning update from real experience.
        4. Store transition in learned model.
        5. Push high-TD-error transition into priority queue.
        6. Run prioritized planning updates from the model.
        7. Propagate value changes backward through predecessors.
    """

    def __init__(
        self,
        action_dim: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        planning_steps: int = 20,
        priority_threshold: float = 1e-4,
        encoder: Optional[StateEncoder] = None,
    ):
        self.action_dim = int(action_dim)

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.planning_steps = int(planning_steps)
        self.priority_threshold = float(priority_threshold)

        self.encoder = encoder if encoder is not None else StateEncoder()

        self.q = TabularQFunction(self.action_dim)
        self.model = ModelMemory()
        self.priority_queue = PriorityQueue()

        self.total_real_updates = 0
        self.total_planning_updates = 0

    def encode(self, obs) -> tuple:
        return self.encoder.encode(obs)

    def select_action(self, obs, training: bool = True) -> int:
        state_key = self.encode(obs)

        action_mask = np.asarray(
            obs["action_mask"],
            dtype=np.float32,
        )

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return 0

        if training and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        return self.q.best_action(
            state_key,
            action_mask,
        )

    def act(self, obs) -> int:
        return self.select_action(
            obs,
            training=False,
        )

    def _target(
        self,
        reward: float,
        next_state_key: tuple,
        done: bool,
        next_action_mask,
    ) -> float:
        if done:
            return float(reward)

        return float(reward) + self.gamma * self.q.max_value(
            next_state_key,
            next_action_mask,
        )

    def compute_priority(
        self,
        state_key: tuple,
        action: int,
        reward: float,
        next_state_key: tuple,
        done: bool,
        next_action_mask,
    ) -> float:
        current_q = self.q.value(
            state_key,
            action,
        )

        target = self._target(
            reward,
            next_state_key,
            done,
            next_action_mask,
        )

        return abs(target - current_q)

    def observe(
        self,
        obs,
        action: int,
        reward: float,
        next_obs,
        done: bool,
    ) -> dict:
        state_key = self.encode(obs)
        next_state_key = self.encode(next_obs)

        action_mask = np.asarray(
            obs["action_mask"],
            dtype=np.float32,
        )

        next_action_mask = np.asarray(
            next_obs["action_mask"],
            dtype=np.float32,
        )

        priority_before_update = self.compute_priority(
            state_key,
            action,
            reward,
            next_state_key,
            done,
            next_action_mask,
        )

        real_td_error = self.q.update(
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
            next_action_mask=next_action_mask,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.total_real_updates += 1

        self.model.store(
            state_key=state_key,
            action=action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
            action_mask=action_mask,
            next_action_mask=next_action_mask,
        )

        if priority_before_update > self.priority_threshold:
            self.priority_queue.push(
                priority_before_update,
                state_key,
                action,
            )

        planning_info = self.planning()

        return {
            "real_td_error": float(real_td_error),
            "planning_updates": planning_info["planning_updates"],
            "mean_planning_td_error": planning_info["mean_planning_td_error"],
            "model_size": len(self.model),
            "queue_size": len(self.priority_queue),
        }

    def planning(self) -> dict:
        planning_td_errors = []
        updates = 0

        for _ in range(self.planning_steps):
            if self.priority_queue.empty():
                break

            priority, state_key, action = self.priority_queue.pop()

            if priority < self.priority_threshold:
                continue

            transition = self.model.get(
                state_key,
                action,
            )

            if transition is None:
                continue

            (
                s_key,
                a,
                reward,
                next_s_key,
                done,
                action_mask,
                next_action_mask,
            ) = transition

            td_error = self.q.update(
                state_key=s_key,
                action=a,
                reward=reward,
                next_state_key=next_s_key,
                done=done,
                next_action_mask=next_action_mask,
                alpha=self.alpha,
                gamma=self.gamma,
            )

            planning_td_errors.append(abs(td_error))
            updates += 1
            self.total_planning_updates += 1

            predecessors = self.model.get_predecessors(s_key)

            for pred_state_key, pred_action in predecessors:
                pred_transition = self.model.get(
                    pred_state_key,
                    pred_action,
                )

                if pred_transition is None:
                    continue

                (
                    ps_key,
                    pa,
                    pr,
                    pns_key,
                    pdone,
                    pmask,
                    pnext_mask,
                ) = pred_transition

                pred_priority = self.compute_priority(
                    ps_key,
                    pa,
                    pr,
                    pns_key,
                    pdone,
                    pnext_mask,
                )

                if pred_priority > self.priority_threshold:
                    self.priority_queue.push(
                        pred_priority,
                        ps_key,
                        pa,
                    )

        if updates == 0:
            mean_td = 0.0
        else:
            mean_td = float(np.mean(planning_td_errors))

        return {
            "planning_updates": updates,
            "mean_planning_td_error": mean_td,
        }

    def decay_epsilon(self) -> None:
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

    def action_values(self, obs):
        state_key = self.encode(obs)
        return self.q.values(state_key).copy()

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)

        if directory != "":
            os.makedirs(directory, exist_ok=True)

        payload = {
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "planning_steps": self.planning_steps,
            "priority_threshold": self.priority_threshold,
            "encoder_params": dict(self.encoder.__dict__),
            "q_table": dict(self.q.table),
            "total_real_updates": self.total_real_updates,
            "total_planning_updates": self.total_planning_updates,
        }

        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)

        encoder = StateEncoder(
            **payload.get("encoder_params", {})
        )

        agent = cls(
            action_dim=int(payload["action_dim"]),
            alpha=float(payload.get("alpha", 0.1)),
            gamma=float(payload.get("gamma", 0.99)),
            epsilon_start=float(payload.get("epsilon", 0.05)),
            epsilon_min=float(payload.get("epsilon_min", 0.05)),
            epsilon_decay=float(payload.get("epsilon_decay", 0.995)),
            planning_steps=int(payload.get("planning_steps", 20)),
            priority_threshold=float(payload.get("priority_threshold", 1e-4)),
            encoder=encoder,
        )

        agent.q.table = defaultdict(
            lambda: np.zeros(agent.action_dim, dtype=np.float32)
        )

        for state_key, values in payload["q_table"].items():
            agent.q.table[state_key] = np.asarray(
                values,
                dtype=np.float32,
            )

        agent.total_real_updates = int(
            payload.get("total_real_updates", 0)
        )

        agent.total_planning_updates = int(
            payload.get("total_planning_updates", 0)
        )

        return agent