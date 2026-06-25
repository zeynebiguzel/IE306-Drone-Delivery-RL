from __future__ import annotations

from pathlib import Path

import numpy as np

from priority_dynaq import PriorityDynaQAgent


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
        Official GreedyNearest warm-start
        + active one-step rollout battery/reassignment planner.

    The policy starts from the greedy_nearest decision and changes it only when
    a simple one-step model detects battery risk. In that case, the planner keeps
    the same order but searches for a safer valid drone. If no safer drone exists,
    it may send the risky drone to charge.

    This is a planning/model-based Role C policy, not a DQN policy.
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
        action_mask = np.asarray(
            obs["action_mask"],
            dtype=np.float32,
        )

        valid_actions = np.flatnonzero(action_mask)

        if len(valid_actions) == 0:
            return 0

        greedy_action = self._call_greedy(obs)

        if greedy_action is None:
            return int(valid_actions[0])

        greedy_action = int(greedy_action)

        if not (0 <= greedy_action < len(action_mask)):
            return int(valid_actions[0])

        if action_mask[greedy_action] <= 0:
            return int(valid_actions[0])

        improved_action = self._battery_guard(
            obs=obs,
            action_mask=action_mask,
            greedy_action=greedy_action,
        )

        if (
            0 <= improved_action < len(action_mask)
            and action_mask[improved_action] > 0
        ):
            return int(improved_action)

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

    def _battery_guard(
        self,
        obs,
        action_mask,
        greedy_action,
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
        )

        if safer_action is not None:
            return int(safer_action)

        charge_action = charge_start + drone_id

        if (
            charge_action < len(action_mask)
            and action_mask[charge_action] > 0
            and soc < 0.16
        ):
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

            score = (
                1.00 * trip
                - 4.00 * soc
            )

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _trip_distance(self, drone, order):
        to_pickup = (
            abs(float(drone[0]) - float(order[0]))
            + abs(float(drone[1]) - float(order[1]))
        )

        delivery = (
            abs(float(order[0]) - float(order[2]))
            + abs(float(order[1]) - float(order[3]))
        )

        return to_pickup + delivery

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