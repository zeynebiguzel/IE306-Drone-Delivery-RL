from __future__ import annotations

import atexit
import gzip
import heapq
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


META_KEEP = 0
META_SWAP = 1
META_CHARGE = 2

META_ACTION_NAMES = {
    META_KEEP: "keep_greedy",
    META_SWAP: "safe_swap",
    META_CHARGE: "charge_risky",
}


DEBUG_COUNTERS = {
    "calls": 0,
    "meta_keep": 0,
    "meta_swap": 0,
    "meta_charge": 0,
    "meta_fallback": 0,
    "q_nonzero_states": 0,
}


def _print_debug_counters():
    print("\n[Meta Dyna-Q Planner Debug]")
    for key, value in DEBUG_COUNTERS.items():
        print(f"{key}: {value}")


atexit.register(_print_debug_counters)


def _bucket(value, step, min_value=None, max_value=None):
    value = float(value)

    if min_value is not None:
        value = max(float(min_value), value)

    if max_value is not None:
        value = min(float(max_value), value)

    if step <= 0:
        return int(value)

    return int(value // step)


def _clip_int(value, low, high):
    return int(max(low, min(high, int(value))))


def _is_live_order(order) -> bool:
    return bool(np.any(np.abs(order) > 1e-6))


def _pickup_distance(drone, order) -> float:
    return (
        abs(float(drone[0]) - float(order[0]))
        + abs(float(drone[1]) - float(order[1]))
    )


def _trip_distance(drone, order) -> float:
    to_pickup = _pickup_distance(drone, order)

    delivery = (
        abs(float(order[0]) - float(order[2]))
        + abs(float(order[1]) - float(order[3]))
    )

    return float(to_pickup + delivery)


def _decode_action(obs, action: int) -> dict:
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)

    n_drones = drones.shape[0]
    k_orders = orders.shape[0]

    assign_end = n_drones * k_orders
    charge_start = assign_end
    charge_end = charge_start + n_drones

    action = int(action)

    if 0 <= action < assign_end:
        return {
            "type": "assign",
            "drone_id": action // k_orders,
            "order_slot": action % k_orders,
        }

    if charge_start <= action < charge_end:
        return {
            "type": "charge",
            "drone_id": action - charge_start,
            "order_slot": None,
        }

    return {
        "type": "noop",
        "drone_id": None,
        "order_slot": None,
    }


def _charge_action_for_drone(obs, drone_id: int) -> int:
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)

    n_drones = drones.shape[0]
    k_orders = orders.shape[0]

    return n_drones * k_orders + int(drone_id)


def _greedy_assignment_context(obs, greedy_action: int) -> dict:
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)

    decoded = _decode_action(obs, greedy_action)

    ctx = {
        "is_assign": False,
        "drone_id": None,
        "order_slot": None,
        "soc": 1.0,
        "trip": 0.0,
        "required_soc": 0.0,
        "risk_margin": 1.0,
        "risk_flag": 0,
        "order_age": 0.0,
    }

    if decoded["type"] != "assign":
        return ctx

    drone_id = decoded["drone_id"]
    order_slot = decoded["order_slot"]

    if drone_id is None or order_slot is None:
        return ctx

    if drone_id >= drones.shape[0] or order_slot >= orders.shape[0]:
        return ctx

    drone = drones[drone_id]
    order = orders[order_slot]

    if not _is_live_order(order):
        return ctx

    soc = float(drone[2])
    trip = _trip_distance(drone, order)
    required_soc = 0.180 + 0.0060 * trip
    risk_margin = soc - required_soc
    risk_flag = 1 if risk_margin < 0.0 else 0

    ctx.update(
        {
            "is_assign": True,
            "drone_id": int(drone_id),
            "order_slot": int(order_slot),
            "soc": float(soc),
            "trip": float(trip),
            "required_soc": float(required_soc),
            "risk_margin": float(risk_margin),
            "risk_flag": int(risk_flag),
            "order_age": float(order[4]),
        }
    )

    return ctx


def find_safe_swap_action(
    obs,
    action_mask,
    greedy_action: int,
):
    drones = np.asarray(obs["drones"], dtype=np.float32)
    orders = np.asarray(obs["orders"], dtype=np.float32)
    action_mask = np.asarray(action_mask, dtype=np.float32)

    ctx = _greedy_assignment_context(obs, greedy_action)

    if not ctx["is_assign"]:
        return None

    forbidden_drone = int(ctx["drone_id"])
    order_slot = int(ctx["order_slot"])
    greedy_trip = float(ctx["trip"])
    greedy_soc = float(ctx["soc"])

    n_drones = drones.shape[0]
    k_orders = orders.shape[0]

    if order_slot >= k_orders:
        return None

    order = orders[order_slot]

    if not _is_live_order(order):
        return None

    best_action = None
    best_score = float("inf")

    for d in range(n_drones):
        if d == forbidden_drone:
            continue

        action = d * k_orders + order_slot

        if action >= len(action_mask):
            continue

        if action_mask[action] <= 0:
            continue

        drone = drones[d]

        soc = float(drone[2])
        trip = _trip_distance(drone, order)

        if trip > greedy_trip + 6.0:
            continue

        if soc < greedy_soc + 0.08:
            continue

        required_soc = 0.120 + 0.0040 * trip

        if soc < required_soc:
            continue

        score = (
            1.00 * trip
            - 4.00 * soc
        )

        if score < best_score:
            best_score = score
            best_action = int(action)

    return best_action

def meta_action_mask(obs, greedy_action: int):
    action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

    mask = np.zeros(3, dtype=np.float32)
    mask[META_KEEP] = 1.0

    ctx = _greedy_assignment_context(obs, greedy_action)

    if not ctx["is_assign"]:
        return mask

    # V2:
    # Allow Meta Dyna-Q to intervene not only when greedy is already risky,
    # but also when it is close to risky.
    is_intervention_zone = (
        int(ctx["risk_flag"]) == 1
        or float(ctx["risk_margin"]) < 0.08
        or float(ctx["soc"]) < 0.32
    )

    if not is_intervention_zone:
        return mask

    safe_swap = find_safe_swap_action(
        obs=obs,
        action_mask=action_mask,
        greedy_action=greedy_action,
    )

    if safe_swap is not None:
        mask[META_SWAP] = 1.0

    drone_id = int(ctx["drone_id"])
    charge_action = _charge_action_for_drone(obs, drone_id)

    # Charge is still stricter than swap, but less strict than before.
    charge_zone = (
        float(ctx["soc"]) < 0.18
        or float(ctx["risk_margin"]) < -0.03
    )

    if (
        charge_zone
        and charge_action < len(action_mask)
        and action_mask[charge_action] > 0
    ):
        mask[META_CHARGE] = 1.0

    return mask

def meta_to_env_action(
    obs,
    greedy_action: int,
    meta_action: int,
):
    action_mask = np.asarray(obs["action_mask"], dtype=np.float32)
    greedy_action = int(greedy_action)
    meta_action = int(meta_action)

    if meta_action == META_KEEP:
        return int(greedy_action), "keep_greedy"

    if meta_action == META_SWAP:
        safe_swap = find_safe_swap_action(
            obs=obs,
            action_mask=action_mask,
            greedy_action=greedy_action,
        )

        if safe_swap is not None:
            return int(safe_swap), "safe_swap"

        return int(greedy_action), "fallback_keep"

    if meta_action == META_CHARGE:
        ctx = _greedy_assignment_context(obs, greedy_action)

        if ctx["is_assign"]:
            drone_id = int(ctx["drone_id"])
            charge_action = _charge_action_for_drone(obs, drone_id)

            if (
                charge_action < len(action_mask)
                and action_mask[charge_action] > 0
            ):
                return int(charge_action), "charge_risky"

        return int(greedy_action), "fallback_keep"

    return int(greedy_action), "fallback_keep"


class MetaStateEncoder:
    """
    Compact state encoder for Meta Dyna-Q.

    Instead of encoding the full drone-dispatch state, this encoder focuses on
    the planning situation around the greedy action:
        - Is greedy assigning an order?
        - Is the selected drone battery-risky?
        - Is there a safer same-order drone?
        - How many live orders / low-battery drones exist?
        - What is the current time bucket?

    This makes Dyna-Q much more likely to generalize.
    """

    def __init__(
        self,
        soc_bins: int = 10,
        trip_bin: int = 5,
        age_bin: int = 10,
        time_bins: int = 10,
    ):
        self.soc_bins = int(soc_bins)
        self.trip_bin = int(trip_bin)
        self.age_bin = int(age_bin)
        self.time_bins = int(time_bins)
    def encode(self, obs, greedy_action: int) -> tuple:
        drones = np.asarray(obs["drones"], dtype=np.float32)
        orders = np.asarray(obs["orders"], dtype=np.float32)
        time = np.asarray(obs["time"], dtype=np.float32)

        decoded = _decode_action(obs, greedy_action)
        ctx = _greedy_assignment_context(obs, greedy_action)
        action_mask = np.asarray(obs["action_mask"], dtype=np.float32)

        live_orders = int(
            sum(1 for o in orders if _is_live_order(o))
        )

        low_soc_drones = int(
            sum(1 for d in drones if float(d[2]) < 0.30)
        )

        very_low_soc_drones = int(
            sum(1 for d in drones if float(d[2]) < 0.18)
        )

        if decoded["type"] == "assign":
            greedy_type = 0
        elif decoded["type"] == "charge":
            greedy_type = 1
        else:
            greedy_type = 2

        safe_swap = find_safe_swap_action(
            obs=obs,
            action_mask=action_mask,
            greedy_action=greedy_action,
        )

        has_safe_swap = 1 if safe_swap is not None else 0

        meta_mask = meta_action_mask(
            obs=obs,
            greedy_action=greedy_action,
        )

        can_charge = int(meta_mask[META_CHARGE] > 0)

        # V2: Coarser buckets.
        # This increases generalization: eval states are more likely to match
        # states seen during training.
        soc_bucket = _clip_int(
            int(np.clip(ctx["soc"], 0.0, 1.0) * 5),
            0,
            5,
        )

        trip_bucket = _clip_int(
            _bucket(ctx["trip"], 10, min_value=0, max_value=100),
            0,
            10,
        )

        age_bucket = _clip_int(
            _bucket(ctx["order_age"], 20, min_value=0, max_value=200),
            0,
            10,
        )

        margin = float(ctx["risk_margin"])

        if margin < -0.08:
            margin_bucket = -3
        elif margin < -0.03:
            margin_bucket = -2
        elif margin < 0.00:
            margin_bucket = -1
        elif margin < 0.03:
            margin_bucket = 0
        elif margin < 0.08:
            margin_bucket = 1
        else:
            margin_bucket = 2

        time_bucket = _clip_int(
            int(np.clip(float(time[0]), 0.0, 1.0) * 5),
            0,
            5,
        )

        live_bucket = _clip_int(live_orders // 3, 0, 8)

        return (
            live_bucket,
            _clip_int(low_soc_drones, 0, 8),
            _clip_int(very_low_soc_drones, 0, 8),
            _clip_int(greedy_type, 0, 2),
            int(ctx["is_assign"]),
            int(ctx["risk_flag"]),
            soc_bucket,
            trip_bucket,
            age_bucket,
            margin_bucket,
            int(has_safe_swap),
            int(can_charge),
            time_bucket,
        )


class TabularMetaQ:
    def __init__(self, action_dim: int = 3):
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
            return META_KEEP

        masked = np.full_like(q_values, -1e9)
        masked[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked))

    def has_nonzero_preference(self, state_key, action_mask) -> bool:
        q_values = self.table[state_key]
        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return False

        vals = q_values[valid_actions]

        return bool(float(np.max(vals) - np.min(vals)) > 1e-6)

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


Transition = Tuple[
    tuple,
    int,
    float,
    tuple,
    bool,
    np.ndarray,
    np.ndarray,
]


class MetaModelMemory:
    """
    Dyna-Q learned model for meta-actions.

    Stores:
        (meta_state, meta_action) -> transition

    This is the model used for planning updates.
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


class MetaPriorityQueue:
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


class MetaDynaQAgent:
    """
    Real Dyna-Q agent over meta-actions.

    Dyna-Q logic:
        1. Observe real transition after executing a meta-action.
        2. Direct Q-learning update.
        3. Store transition in learned model.
        4. Push high TD-error transition into priority queue.
        5. Run planning updates from model memory.
        6. Propagate value changes through predecessor states.
    """

    def __init__(
        self,
        alpha: float = 0.10,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        planning_steps: int = 20,
        priority_threshold: float = 1e-4,
        encoder: Optional[MetaStateEncoder] = None,
    ):
        self.action_dim = 3

        self.alpha = float(alpha)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon_start)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.planning_steps = int(planning_steps)
        self.priority_threshold = float(priority_threshold)

        self.encoder = encoder if encoder is not None else MetaStateEncoder()

        self.q = TabularMetaQ(self.action_dim)
        self.model = MetaModelMemory()
        self.priority_queue = MetaPriorityQueue()

        self.total_real_updates = 0
        self.total_planning_updates = 0

    def encode(self, obs, greedy_action: int) -> tuple:
        return self.encoder.encode(
            obs=obs,
            greedy_action=greedy_action,
        )
    def select_meta_action(
        self,
        obs,
        greedy_action: int,
        training: bool = True,
    ) -> int:
        state_key = self.encode(
            obs=obs,
            greedy_action=greedy_action,
        )

        mask = meta_action_mask(
            obs=obs,
            greedy_action=greedy_action,
        )

        valid_actions = np.flatnonzero(mask)

        if len(valid_actions) == 0:
            return META_KEEP

        if training and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        q_values = self.q.values(state_key).copy()

        # V2 evaluation-time intervention priority.
        # This does not remove Dyna-Q; it biases learned Q-values toward
        # intervention only in risky states.
        if not training:
            ctx = _greedy_assignment_context(
                obs=obs,
                greedy_action=greedy_action,
            )

            if ctx["is_assign"]:
                margin = float(ctx["risk_margin"])
                soc = float(ctx["soc"])

                if mask[META_SWAP] > 0:
                    if margin < 0.08:
                        q_values[META_SWAP] += 0.20
                    if margin < 0.03:
                        q_values[META_SWAP] += 0.35
                    if margin < 0.00:
                        q_values[META_SWAP] += 0.60

                if mask[META_CHARGE] > 0:
                    if soc < 0.18 or margin < -0.03:
                        q_values[META_CHARGE] += 0.30
                    if soc < 0.13 or margin < -0.08:
                        q_values[META_CHARGE] += 0.65

        masked = np.full_like(q_values, -1e9)
        masked[valid_actions] = q_values[valid_actions]

        return int(np.argmax(masked))

    def has_nonzero_preference(self, obs, greedy_action: int) -> bool:
        state_key = self.encode(
            obs=obs,
            greedy_action=greedy_action,
        )

        mask = meta_action_mask(
            obs=obs,
            greedy_action=greedy_action,
        )

        return self.q.has_nonzero_preference(
            state_key=state_key,
            action_mask=mask,
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

        priority = abs(target - current_q)

        # V2 prioritized sweeping:
        # Give intervention actions larger priority so planning focuses more
        # on swap/charge decisions instead of mostly keep_greedy.
        if int(action) == META_SWAP:
            priority *= 2.50
        elif int(action) == META_CHARGE:
            priority *= 2.00

        return float(priority)

    def observe(
        self,
        obs,
        greedy_action: int,
        meta_action: int,
        reward: float,
        next_obs,
        next_greedy_action: int,
        done: bool,
    ) -> dict:
        state_key = self.encode(
            obs=obs,
            greedy_action=greedy_action,
        )

        next_state_key = self.encode(
            obs=next_obs,
            greedy_action=next_greedy_action,
        )

        mask = meta_action_mask(
            obs=obs,
            greedy_action=greedy_action,
        )

        next_mask = meta_action_mask(
            obs=next_obs,
            greedy_action=next_greedy_action,
        )

        priority_before_update = self.compute_priority(
            state_key=state_key,
            action=meta_action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
            next_action_mask=next_mask,
        )

        real_td_error = self.q.update(
            state_key=state_key,
            action=meta_action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
            next_action_mask=next_mask,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        self.total_real_updates += 1

        self.model.store(
            state_key=state_key,
            action=meta_action,
            reward=reward,
            next_state_key=next_state_key,
            done=done,
            action_mask=mask,
            next_action_mask=next_mask,
        )

        if priority_before_update > self.priority_threshold:
            self.priority_queue.push(
                priority_before_update,
                state_key,
                meta_action,
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
                    state_key=ps_key,
                    action=pa,
                    reward=pr,
                    next_state_key=pns_key,
                    done=pdone,
                    next_action_mask=pnext_mask,
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

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)

        if directory != "":
            os.makedirs(directory, exist_ok=True)

        payload = {
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

        encoder = MetaStateEncoder(
            **payload.get("encoder_params", {})
        )

        agent = cls(
            alpha=float(payload.get("alpha", 0.10)),
            gamma=float(payload.get("gamma", 0.99)),
            epsilon_start=float(payload.get("epsilon", 0.02)),
            epsilon_min=float(payload.get("epsilon_min", 0.02)),
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


class TrainedMetaDynaQPlanner:
    """
    Final Role C policy:
        Meta Dyna-Q Battery-Reassignment Planner.

    The policy uses GreedyNearest as a warm-start, but the final correction
    decision is selected by a trained Dyna-Q meta-controller.

    Meta-actions:
        0 -> keep GreedyNearest action
        1 -> apply safe same-order drone swap
        2 -> send risky drone to charge
    """

    def __init__(
        self,
        weights_path=None,
    ):
        self.greedy = self._load_official_greedy()
        self.agent = None

        if weights_path is not None:
            path = self._resolve_weights_path(weights_path)

            if path is not None:
                self.agent = MetaDynaQAgent.load(path)
                self.agent.epsilon = 0.0

    def _load_official_greedy(self):
        try:
            from drone_dispatch_env import Config
            from drone_dispatch_env.baselines import GreedyNearest

            root = Path(__file__).resolve().parents[2]
            cfg_path = root / "configs" / "eval_standard.yaml"

            if cfg_path.exists():
                cfg = Config.from_yaml(str(cfg_path))
            else:
                cfg = Config()

            return GreedyNearest(cfg)

        except Exception as e:
            raise RuntimeError(
                f"Official GreedyNearest could not be loaded: {e}"
            )

    def _resolve_weights_path(self, weights_path):
        path = Path(weights_path)

        candidates = [
            path,
            Path(str(path) + ".gz"),
            Path(str(path).replace(".pt", ".pkl.gz")),
            Path(str(path).replace(".pt", ".gz")),
        ]

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return None

    def _call_greedy(self, obs):
        candidates = []

        if hasattr(self.greedy, "act"):
            candidates.append(self.greedy.act)

        if hasattr(self.greedy, "select_action"):
            candidates.append(self.greedy.select_action)

        if hasattr(self.greedy, "predict"):
            candidates.append(self.greedy.predict)

        if callable(self.greedy):
            candidates.append(self.greedy)

        for fn in candidates:
            for args in [
                (obs,),
                (obs, None),
                (obs, {}),
            ]:
                try:
                    out = fn(*args)

                    if isinstance(out, tuple):
                        out = out[0]

                    return int(out)

                except TypeError:
                    continue

                except Exception:
                    continue

        return None

    def act(self, obs):
        DEBUG_COUNTERS["calls"] += 1

        action_mask = np.asarray(
            obs["action_mask"],
            dtype=np.float32,
        )

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return 0

        greedy_action = self._call_greedy(obs)

        if greedy_action is None:
            greedy_action = int(valid_actions[0])

        greedy_action = int(greedy_action)

        if not (0 <= greedy_action < len(action_mask)):
            greedy_action = int(valid_actions[0])

        if action_mask[greedy_action] <= 0:
            greedy_action = int(valid_actions[0])

        if self.agent is None:
            DEBUG_COUNTERS["meta_fallback"] += 1
            return int(greedy_action)

        if self.agent.has_nonzero_preference(
            obs=obs,
            greedy_action=greedy_action,
        ):
            DEBUG_COUNTERS["q_nonzero_states"] += 1

        meta_action = self.agent.select_meta_action(
            obs=obs,
            greedy_action=greedy_action,
            training=False,
        )

        env_action, reason = meta_to_env_action(
            obs=obs,
            greedy_action=greedy_action,
            meta_action=meta_action,
        )

        if reason == "keep_greedy":
            DEBUG_COUNTERS["meta_keep"] += 1
        elif reason == "safe_swap":
            DEBUG_COUNTERS["meta_swap"] += 1
        elif reason == "charge_risky":
            DEBUG_COUNTERS["meta_charge"] += 1
        else:
            DEBUG_COUNTERS["meta_fallback"] += 1

        if (
            0 <= env_action < len(action_mask)
            and action_mask[env_action] > 0
        ):
            return int(env_action)

        DEBUG_COUNTERS["meta_fallback"] += 1
        return int(greedy_action)

    def action_values(self, obs):
        if self.agent is None:
            return None

        greedy_action = self._call_greedy(obs)

        if greedy_action is None:
            return None

        state_key = self.agent.encode(
            obs=obs,
            greedy_action=int(greedy_action),
        )

        return self.agent.q.values(state_key).copy()

    def action_probs(self, obs):
        return None

    def state_values(self, obs):
        return None