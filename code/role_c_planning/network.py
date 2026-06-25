from __future__ import annotations

import gzip
import pickle
from collections import defaultdict

import numpy as np


def _safe_int(value) -> int:
    return int(float(value))


def _bucket(value, step, max_bucket=None) -> int:
    if step <= 0:
        out = int(value)
    else:
        out = int(float(value) // step)

    if max_bucket is not None:
        out = max(0, min(int(max_bucket), out))

    return out


class StateEncoder:
    """
    Discretizes DroneDispatch-v0 observations for tabular Dyna-Q.

    This is NOT a neural network.
    Role C is planning/model-based acceleration, so we use a compact state
    abstraction and a table-based Q function.
    """

    def __init__(
        self,
        position_bin: int = 2,
        order_position_bin: int = 3,
        soc_bins: int = 5,
        age_bin: int = 10,
        time_bins: int = 10,
        top_k_orders: int = 8,
    ):
        self.position_bin = int(position_bin)
        self.order_position_bin = int(order_position_bin)
        self.soc_bins = int(soc_bins)
        self.age_bin = int(age_bin)
        self.time_bins = int(time_bins)
        self.top_k_orders = int(top_k_orders)

    def encode(self, obs) -> tuple:
        drones = np.asarray(obs["drones"], dtype=np.float32)
        orders = np.asarray(obs["orders"], dtype=np.float32)
        time = np.asarray(obs["time"], dtype=np.float32)

        drone_part = self._encode_drones(drones)
        order_part = self._encode_orders(orders)
        summary_part = self._encode_summary(drones, orders, time)

        return tuple(summary_part + drone_part + order_part)

    def _encode_drones(self, drones) -> list[int]:
        encoded = []

        for d in drones:
            x = _bucket(d[0], self.position_bin)
            y = _bucket(d[1], self.position_bin)

            soc = int(np.clip(d[2], 0.0, 1.0) * self.soc_bins)
            soc = min(self.soc_bins, soc)

            alive = 1 if d[3] > 0.5 else 0

            status_onehot = d[4:9]
            status = int(np.argmax(status_onehot))

            has_order = 1 if d[9] > 0.5 else 0

            encoded.extend(
                [
                    x,
                    y,
                    soc,
                    alive,
                    status,
                    has_order,
                ]
            )

        return encoded

    def _encode_orders(self, orders) -> list[int]:
        encoded = []

        k = min(self.top_k_orders, len(orders))

        for i in range(k):
            o = orders[i]

            is_live = 1 if np.any(np.abs(o) > 1e-6) else 0

            if not is_live:
                encoded.extend([0, 0, 0, 0, 0, 0])
                continue

            ox = _bucket(o[0], self.order_position_bin)
            oy = _bucket(o[1], self.order_position_bin)
            dx = _bucket(o[2], self.order_position_bin)
            dy = _bucket(o[3], self.order_position_bin)
            age = _bucket(o[4], self.age_bin, max_bucket=20)

            encoded.extend(
                [
                    is_live,
                    ox,
                    oy,
                    dx,
                    dy,
                    age,
                ]
            )

        missing = self.top_k_orders - k

        for _ in range(missing):
            encoded.extend([0, 0, 0, 0, 0, 0])

        return encoded

    def _encode_summary(self, drones, orders, time) -> list[int]:
        live_orders = int(
            sum(1 for o in orders if np.any(np.abs(o) > 1e-6))
        )

        idle_drones = 0
        low_soc_drones = 0
        carrying_drones = 0
        alive_drones = 0

        for d in drones:
            alive = d[3] > 0.5
            status = int(np.argmax(d[4:9]))

            if alive:
                alive_drones += 1

            if alive and status == 0:
                idle_drones += 1

            if alive and d[2] < 0.35:
                low_soc_drones += 1

            if alive and d[9] > 0.5:
                carrying_drones += 1

        time_bucket = int(np.clip(time[0], 0.0, 1.0) * self.time_bins)

        return [
            live_orders,
            idle_drones,
            low_soc_drones,
            carrying_drones,
            alive_drones,
            time_bucket,
        ]


class TabularQFunction:
    """
    Table-based Q(s,a) storage for Dyna-Q.
    """

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)
        self.table = defaultdict(
            lambda: np.zeros(self.action_dim, dtype=np.float32)
        )

    def values(self, state_key) -> np.ndarray:
        return self.table[state_key]

    def value(self, state_key, action: int) -> float:
        return float(self.table[state_key][int(action)])

    def max_value(self, state_key, action_mask=None) -> float:
        q_values = self.table[state_key]

        if action_mask is None:
            return float(np.max(q_values))

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return 0.0

        return float(np.max(q_values[valid_actions]))

    def best_action(self, state_key, action_mask) -> int:
        q_values = self.table[state_key].copy()

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return 0

        masked = np.full_like(q_values, -1e9)
        masked[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked))

    def update(
        self,
        state_key,
        action: int,
        reward: float,
        next_state_key,
        done: bool,
        next_action_mask,
        alpha: float,
        gamma: float,
    ) -> float:
        action = int(action)

        current_q = self.table[state_key][action]

        if done:
            target = float(reward)
        else:
            target = float(reward) + gamma * self.max_value(
                next_state_key,
                next_action_mask,
            )

        td_error = target - current_q

        self.table[state_key][action] += alpha * td_error

        return float(td_error)

    def save(self, path: str) -> None:
        payload = {
            "action_dim": self.action_dim,
            "table": dict(self.table),
        }

        with gzip.open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str):
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)

        obj = cls(payload["action_dim"])

        for key, values in payload["table"].items():
            obj.table[key] = np.asarray(values, dtype=np.float32)

        return obj