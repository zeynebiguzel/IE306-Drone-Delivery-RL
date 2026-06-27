from __future__ import annotations

import atexit
from pathlib import Path

import numpy as np

from priority_dynaq import PriorityDynaQAgent


DEBUG_COUNTERS = {
    "calls": 0,
    "dynaq_q_available": 0,
    "dynaq_proposals": 0,
    "dynaq_accepted": 0,
    "guard_swaps": 0,
    "guard_charges": 0,
    "kept_greedy": 0,
}


def _print_debug_counters():
    print("\n[Role C Dyna-Q Assisted Planner Debug]")
    for key, value in DEBUG_COUNTERS.items():
        print(f"{key}: {value}")


atexit.register(_print_debug_counters)


def preprocess_state(obs):
    drones = np.asarray(obs["drones"], dtype=np.float32).flatten()
    orders = np.asarray(obs["orders"], dtype=np.float32).flatten()
    grid = np.asarray(obs["grid"], dtype=np.float32).flatten()
    time = np.asarray(obs["time"], dtype=np.float32).flatten()

    return np.concatenate(
        [
            drones,
            orders,
            grid,
            time,
        ]
    ).astype(np.float32)


class TrainedPriorityDynaQPolicy:
    """
    Role C policy:
        Dyna-Q assisted one-step rollout planner.

    Decision flow:
        1. Load trained Priority Dyna-Q Q-table.
        2. Get GreedyNearest action as a strong warm-start.
        3. Let Dyna-Q propose actions from the learned Q-table.
        4. Accept Dyna-Q proposal only if it is valid and safe.
        5. Otherwise apply one-step battery-risk guard on GreedyNearest.

    This makes Dyna-Q part of the deployed decision mechanism, while the
    rollout guard prevents unsafe Q-table actions from damaging performance.
    """

    def __init__(
        self,
        state_size,
        action_size,
        weights_path=None,
    ):
        self.state_size = int(state_size)
        self.action_size = int(action_size)

        self.agent = None
        self.greedy = self._load_official_greedy()

        if weights_path is not None:
            path = self._resolve_weights_path(weights_path)

            if path is not None:
                try:
                    self.agent = PriorityDynaQAgent.load(path)
                    self.agent.epsilon = 0.0
                except Exception:
                    self.agent = None

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

        q_values = self._get_dynaq_values(obs, action_mask)

        dynaq_action = self._dynaq_safe_proposal(
            obs=obs,
            action_mask=action_mask,
            greedy_action=greedy_action,
            q_values=q_values,
        )

        if dynaq_action is not None:
            DEBUG_COUNTERS["dynaq_accepted"] += 1
            return int(dynaq_action)

        improved_action = self._battery_guard(
            obs=obs,
            action_mask=action_mask,
            greedy_action=greedy_action,
            q_values=q_values,
        )

        if improved_action != greedy_action:
            return int(improved_action)

        DEBUG_COUNTERS["kept_greedy"] += 1
        return int(greedy_action)

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

    def _get_dynaq_values(self, obs, action_mask):
        if self.agent is None:
            return None

        try:
            q_values = self.agent.action_values(obs)
        except Exception:
            return None

        if q_values is None:
            return None

        q_values = np.asarray(q_values, dtype=np.float32)

        if q_values.shape[0] != len(action_mask):
            return None

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return None

        valid_q = q_values[valid_actions]

        # If all Q-values are almost identical, the Q-table gives no useful preference.
        if float(np.max(valid_q) - np.min(valid_q)) < 1e-6:
            return None

        DEBUG_COUNTERS["dynaq_q_available"] += 1
        return q_values

    def _dynaq_safe_proposal(
        self,
        obs,
        action_mask,
        greedy_action,
        q_values,
    ):
        if q_values is None:
            return None

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return None

        masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
        masked_q[valid_actions] = q_values[valid_actions]

        # Try top Dyna-Q proposals, but accept only safe ones.
        top_k = min(12, len(valid_actions))
        candidate_actions = np.argsort(masked_q)[-top_k:][::-1]

        for candidate in candidate_actions:
            candidate = int(candidate)

            if candidate == greedy_action:
                continue

            if not (0 <= candidate < len(action_mask)):
                continue

            if action_mask[candidate] <= 0:
                continue

            DEBUG_COUNTERS["dynaq_proposals"] += 1

            if self._is_safe_dynaq_candidate(
                obs=obs,
                candidate_action=candidate,
                greedy_action=greedy_action,
            ):
                return candidate

        return None

    def _is_safe_dynaq_candidate(
        self,
        obs,
        candidate_action,
        greedy_action,
    ):
        drones = np.asarray(obs["drones"], dtype=np.float32)
        orders = np.asarray(obs["orders"], dtype=np.float32)

        n_drones = drones.shape[0]
        k_orders = orders.shape[0]
        assign_end = n_drones * k_orders
        charge_start = assign_end
        charge_end = charge_start + n_drones

        # Accept Dyna-Q assignment only if it is a safe same-order improvement
        # or a very safe nearby alternative.
        if candidate_action < assign_end:
            cand_drone_id = candidate_action // k_orders
            cand_order_slot = candidate_action % k_orders

            if cand_drone_id >= n_drones or cand_order_slot >= k_orders:
                return False

            cand_drone = drones[cand_drone_id]
            cand_order = orders[cand_order_slot]

            if not self._is_live_order(cand_order):
                return False

            cand_soc = float(cand_drone[2])
            cand_trip = self._trip_distance(cand_drone, cand_order)
            cand_required = 0.140 + 0.0045 * cand_trip

            if cand_soc < cand_required:
                return False

            # If greedy is assignment, compare with it.
            if greedy_action < assign_end:
                greedy_drone_id = greedy_action // k_orders
                greedy_order_slot = greedy_action % k_orders

                if greedy_drone_id >= n_drones or greedy_order_slot >= k_orders:
                    return False

                greedy_drone = drones[greedy_drone_id]
                greedy_order = orders[greedy_order_slot]

                if not self._is_live_order(greedy_order):
                    return False

                greedy_soc = float(greedy_drone[2])
                greedy_trip = self._trip_distance(greedy_drone, greedy_order)

                # Most conservative case: Dyna-Q chooses another drone for same order.
                if cand_order_slot == greedy_order_slot:
                    if cand_trip <= greedy_trip + 6.0 and cand_soc >= greedy_soc + 0.06:
                        return True

                    return False

                # Different order is riskier. Accept only if it is very close,
                # safe, and older than the greedy order.
                cand_age = float(cand_order[4])
                greedy_age = float(greedy_order[4])

                cand_pickup = self._pickup_distance(cand_drone, cand_order)
                greedy_pickup = self._pickup_distance(greedy_drone, greedy_order)

                if (
                    cand_age >= greedy_age + 10.0
                    and cand_pickup <= greedy_pickup + 2.0
                    and cand_soc >= 0.45
                    and cand_trip <= greedy_trip + 4.0
                ):
                    return True

                return False

            # If greedy is not assignment, accept only very safe assignment.
            return cand_soc >= 0.50

        # Accept Dyna-Q charge only for a low-battery drone.
        if charge_start <= candidate_action < charge_end:
            drone_id = candidate_action - charge_start

            if drone_id >= n_drones:
                return False

            soc = float(drones[drone_id][2])

            if soc < 0.12:
                return True

            return False

        return False

    def _battery_guard(
        self,
        obs,
        action_mask,
        greedy_action,
        q_values=None,
    ):
        drones = np.asarray(obs["drones"], dtype=np.float32)
        orders = np.asarray(obs["orders"], dtype=np.float32)

        n_drones = drones.shape[0]
        k_orders = orders.shape[0]

        assign_end = n_drones * k_orders
        charge_start = assign_end

        # Greedy is charge/noop, keep it.
        if greedy_action >= assign_end:
            return int(greedy_action)

        drone_id = greedy_action // k_orders
        order_slot = greedy_action % k_orders

        if drone_id >= n_drones or order_slot >= k_orders:
            return int(greedy_action)

        drone = drones[drone_id]
        order = orders[order_slot]

        if not self._is_live_order(order):
            return int(greedy_action)

        soc = float(drone[2])
        greedy_trip = self._trip_distance(drone, order)

        required_soc = 0.180 + 0.0060 * greedy_trip

        if soc >= required_soc:
            return int(greedy_action)

        safer_action = self._best_same_order_safe_drone(
            obs=obs,
            action_mask=action_mask,
            order_slot=order_slot,
            forbidden_drone=drone_id,
            greedy_trip=greedy_trip,
            greedy_soc=soc,
            q_values=q_values,
        )

        if safer_action is not None:
            DEBUG_COUNTERS["guard_swaps"] += 1
            return int(safer_action)

        charge_action = charge_start + drone_id

        if (
            charge_action < len(action_mask)
            and action_mask[charge_action] > 0
            and soc < 0.16
        ):
            DEBUG_COUNTERS["guard_charges"] += 1
            return int(charge_action)

        return int(greedy_action)

    def _best_same_order_safe_drone(
        self,
        obs,
        action_mask,
        order_slot,
        forbidden_drone,
        greedy_trip,
        greedy_soc,
        q_values=None,
    ):
        drones = np.asarray(obs["drones"], dtype=np.float32)
        orders = np.asarray(obs["orders"], dtype=np.float32)

        n_drones = drones.shape[0]
        k_orders = orders.shape[0]

        if order_slot >= k_orders:
            return None

        order = orders[order_slot]

        if not self._is_live_order(order):
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
            trip = self._trip_distance(drone, order)

            if trip > greedy_trip + 6.0:
                continue

            if soc < greedy_soc + 0.08:
                continue

            required_soc = 0.120 + 0.0040 * trip

            if soc < required_soc:
                continue

            q_bonus = 0.0

            if q_values is not None and action < len(q_values):
                q_bonus = float(q_values[action])

            # Dyna-Q is used here as a learned tie-break/preference term.
            # The main hard constraints keep the selected action safe.
            score = (
                1.00 * trip
                - 4.00 * soc
                - 0.02 * q_bonus
            )

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _trip_distance(self, drone, order):
        to_pickup = self._pickup_distance(drone, order)

        delivery = (
            abs(float(order[0]) - float(order[2]))
            + abs(float(order[1]) - float(order[3]))
        )

        return to_pickup + delivery

    def _pickup_distance(self, drone, order):
        return (
            abs(float(drone[0]) - float(order[0]))
            + abs(float(drone[1]) - float(order[1]))
        )

    def _is_live_order(self, order):
        return bool(np.any(np.abs(order) > 1e-6))

    def action_values(self, obs):
        if self.agent is None:
            return None

        try:
            return self.agent.action_values(obs)
        except Exception:
            return None

    def action_probs(self, obs):
        return None

    def state_values(self, obs):
        return None