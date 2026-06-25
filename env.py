"""Simulation environment for AVAL retail scenarios."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from catalog import ScenarioSpec, get_scenario


MIN_PRICE = 0.01


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info


class RetailEnv:
    """Small deterministic/stochastic environment for MVP scenario transitions."""

    def __init__(
        self,
        scenario: ScenarioSpec | str,
        seed: int | None = None,
        stochastic: bool = True,
    ) -> None:
        self.scenario = get_scenario(scenario)
        self.rng = np.random.default_rng(seed)
        self.stochastic = stochastic
        self.sku_ids = tuple(sku.sku_id for sku in self.scenario.skus)
        self.bundle_ids = tuple(bundle.bundle_id for bundle in self.scenario.bundles)
        self.offer_ids = self.scenario.offer_ids(include_bundles=True)
        self.n_skus = len(self.sku_ids)
        self.n_bundles = len(self.bundle_ids)
        self.max_shelf_life = max(sku.shelf_life_days for sku in self.scenario.skus)
        self.matrices = self.scenario.to_matrices(include_bundles=True)
        self.bundle_component_matrix = self.matrices["bundle_component_matrix"].astype(
            float
        )
        self.reset()

    def reset(
        self,
        initial_inventory: Sequence[float] | Mapping[str, float] | None = None,
        initial_day: int = 0,
    ) -> dict[str, Any]:
        if initial_day < 0:
            raise ValueError("initial_day cannot be negative.")
        if initial_day > self.scenario.horizon_days:
            raise ValueError("initial_day cannot exceed scenario horizon.")

        self.day = initial_day
        self.current_prices = self.matrices["initial_prices"].astype(float).copy()
        self.cash = self.scenario.starting_cash
        self.min_cash = self.cash
        self.bankrupt = False
        self.inventory_by_age = np.zeros(
            (self.n_skus, self.max_shelf_life), dtype=float
        )
        if initial_inventory is not None:
            inventory = self._coerce_sku_vector(initial_inventory, np.zeros(self.n_skus))
            self.inventory_by_age[:, 0] = np.maximum(0.0, inventory)
        return self.observation()

    def observation(self) -> dict[str, Any]:
        intercepts, slopes = self._demand_params_at_day(self.day)
        expected_demand = self.expected_demand(self.current_prices, day=self.day)
        return {
            "scenario_id": self.scenario.scenario_id,
            "engine": self.scenario.engine,
            "day": self.day,
            "horizon_days": self.scenario.horizon_days,
            "sku_ids": self.sku_ids,
            "bundle_ids": self.bundle_ids,
            "offer_ids": self.offer_ids,
            "costs": self.matrices["costs"].copy(),
            "current_prices": self.current_prices.copy(),
            "inventory": self.inventory_by_age.sum(axis=1).copy(),
            "inventory_by_age": self.inventory_by_age.copy(),
            "shelf_life_days": self.matrices["shelf_life_days"][: self.n_skus].copy(),
            "demand_intercepts": intercepts.copy(),
            "demand_slopes": slopes.copy(),
            "expected_demand": expected_demand,
            "bundle_component_matrix": self.bundle_component_matrix.copy(),
            "starting_cash": self.scenario.starting_cash,
            "cash": self.cash,
            "bankrupt": self.bankrupt,
        }

    def step(self, action: Mapping[str, Any] | None = None) -> StepResult:
        if self.day >= self.scenario.horizon_days:
            raise RuntimeError("Cannot step an environment after it is done.")
        action = {} if action is None else dict(action)

        prices = self._prices_from_action(action)
        orders_requested = self._coerce_sku_vector(
            action.get("orders"), np.zeros(self.n_skus)
        )
        orders = self._clip_orders(orders_requested)
        self._receive_orders(orders)

        expected_demand = self.expected_demand(prices, day=self.day)
        realized_demand = self._realized_demand(expected_demand)
        bundle_sales = self._sell_bundles(realized_demand[self.n_skus :])
        sku_sales = self._sell_skus(realized_demand[: self.n_skus])
        offer_sales = np.concatenate([sku_sales, bundle_sales])

        revenue = float(np.dot(prices, offer_sales))
        order_cost = float(np.dot(self._sku_costs(), orders))
        expired_units, salvage_revenue = self._age_and_expire()
        reward = revenue + salvage_revenue - order_cost

        if self.cash is not None:
            self.cash += reward
            self.min_cash = min(self.min_cash, self.cash)
            self.bankrupt = self.bankrupt or self.cash < 0.0

        self.current_prices = prices
        self.day += 1
        done = self.day >= self.scenario.horizon_days
        observation = self.observation()
        lost_sales = np.maximum(0.0, realized_demand - offer_sales)
        info = {
            "prices": prices.copy(),
            "orders_requested": orders_requested,
            "orders": orders.copy(),
            "expected_demand": expected_demand,
            "realized_demand": realized_demand,
            "offer_sales": offer_sales,
            "sku_sales": sku_sales,
            "bundle_sales": bundle_sales,
            "revenue": revenue,
            "order_cost": order_cost,
            "expired_units": expired_units,
            "salvage_revenue": salvage_revenue,
            "profit": reward,
            "lost_sales": lost_sales,
            "inventory_end": observation["inventory"],
            "cash": self.cash,
            "bankrupt": self.bankrupt,
        }
        return StepResult(observation, reward, done, info)

    def expected_demand(
        self, prices: Sequence[float] | np.ndarray, day: int | None = None
    ) -> np.ndarray:
        day = self.day if day is None else day
        if day < 0:
            raise ValueError("day cannot be negative.")
        prices_array = np.asarray(prices, dtype=float)
        if prices_array.shape != (len(self.offer_ids),):
            raise ValueError(
                f"Expected {len(self.offer_ids)} prices, received {prices_array.size}."
            )
        intercepts, slopes = self._demand_params_at_day(day)
        return np.maximum(0.0, intercepts - slopes * prices_array)

    def _prices_from_action(self, action: Mapping[str, Any]) -> np.ndarray:
        prices = self.current_prices.copy()
        if "markdowns" in action and action["markdowns"] is not None:
            markdowns = self._coerce_offer_vector(
                action["markdowns"], np.zeros(len(self.offer_ids))
            )
            if np.any(markdowns < 0.0) or np.any(markdowns >= 1.0):
                raise ValueError("markdowns must be in the [0, 1) interval.")
            prices = prices * (1.0 - markdowns)
        if "prices" in action and action["prices"] is not None:
            prices = self._coerce_offer_vector(action["prices"], prices)
        if np.any(prices <= 0.0):
            raise ValueError("All prices must be positive.")
        return np.maximum(MIN_PRICE, prices)

    def _coerce_offer_vector(
        self, value: Any, default: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self._coerce_vector(value, default, self.offer_ids, "offer")

    def _coerce_sku_vector(
        self, value: Any, default: np.ndarray | Sequence[float]
    ) -> np.ndarray:
        return self._coerce_vector(value, default, self.sku_ids, "SKU")

    @staticmethod
    def _coerce_vector(
        value: Any,
        default: np.ndarray | Sequence[float],
        ids: Sequence[str],
        label: str,
    ) -> np.ndarray:
        result = np.asarray(default, dtype=float).copy()
        if result.shape != (len(ids),):
            raise ValueError(f"default {label} vector has invalid shape.")
        if value is None:
            return result
        if isinstance(value, Mapping):
            unknown = set(value) - set(ids)
            if unknown:
                unknown_text = ", ".join(sorted(str(item) for item in unknown))
                raise KeyError(f"Unknown {label} ids: {unknown_text}.")
            id_to_index = {item_id: idx for idx, item_id in enumerate(ids)}
            for item_id, item_value in value.items():
                result[id_to_index[item_id]] = float(item_value)
            return result
        array = np.asarray(value, dtype=float)
        if array.shape != (len(ids),):
            raise ValueError(f"Expected {len(ids)} {label} values, received {array.size}.")
        return array.copy()

    def _clip_orders(self, requested: np.ndarray) -> np.ndarray:
        orders = np.maximum(0.0, requested.astype(float))
        max_orders = self.matrices["max_order_units"][: self.n_skus].astype(float)
        has_limit = max_orders > 0.0
        orders[has_limit] = np.minimum(orders[has_limit], max_orders[has_limit])
        if self.scenario.capacity_units is None:
            return orders

        remaining_capacity = self.scenario.capacity_units - float(
            self.inventory_by_age.sum()
        )
        remaining_capacity = max(0.0, remaining_capacity)
        clipped = np.zeros_like(orders)
        for idx, requested_units in enumerate(orders):
            accepted = min(float(requested_units), remaining_capacity)
            clipped[idx] = accepted
            remaining_capacity -= accepted
            if remaining_capacity <= 0.0:
                break
        return clipped

    def _receive_orders(self, orders: np.ndarray) -> None:
        self.inventory_by_age[:, 0] += orders

    def _sell_bundles(self, bundle_demand: np.ndarray) -> np.ndarray:
        bundle_sales = np.zeros(self.n_bundles, dtype=float)
        for bundle_idx, desired_units in enumerate(bundle_demand):
            components = self.bundle_component_matrix[bundle_idx]
            positive = components > 0.0
            if not np.any(positive):
                continue
            inventory = self.inventory_by_age.sum(axis=1)
            component_capacity = np.min(inventory[positive] / components[positive])
            sold = min(float(desired_units), float(component_capacity))
            sold = max(0.0, sold)
            if sold > 0.0:
                for sku_idx, component_qty in enumerate(components):
                    if component_qty > 0.0:
                        self._consume_sku_units(sku_idx, component_qty * sold)
            bundle_sales[bundle_idx] = sold
        return bundle_sales

    def _sell_skus(self, sku_demand: np.ndarray) -> np.ndarray:
        sku_sales = np.zeros(self.n_skus, dtype=float)
        for sku_idx, desired_units in enumerate(sku_demand):
            sold = min(float(desired_units), float(self.inventory_by_age[sku_idx].sum()))
            sold = max(0.0, sold)
            if sold > 0.0:
                self._consume_sku_units(sku_idx, sold)
            sku_sales[sku_idx] = sold
        return sku_sales

    def _consume_sku_units(self, sku_idx: int, requested_units: float) -> float:
        remaining = max(0.0, float(requested_units))
        consumed = 0.0
        shelf_life = self.scenario.skus[sku_idx].shelf_life_days
        for age_idx in range(shelf_life - 1, -1, -1):
            available = self.inventory_by_age[sku_idx, age_idx]
            if available <= 0.0:
                continue
            used = min(float(available), remaining)
            self.inventory_by_age[sku_idx, age_idx] -= used
            remaining -= used
            consumed += used
            if remaining <= 1e-12:
                break
        return consumed

    def _age_and_expire(self) -> tuple[np.ndarray, float]:
        expired = np.zeros(self.n_skus, dtype=float)
        next_inventory = np.zeros_like(self.inventory_by_age)
        for sku_idx, sku in enumerate(self.scenario.skus):
            shelf_life = sku.shelf_life_days
            expired[sku_idx] = float(self.inventory_by_age[sku_idx, shelf_life - 1 :].sum())
            if shelf_life > 1:
                next_inventory[sku_idx, 1:shelf_life] = self.inventory_by_age[
                    sku_idx, : shelf_life - 1
                ]
        self.inventory_by_age = next_inventory
        salvage_values = self.matrices["salvage_values"][: self.n_skus]
        salvage_revenue = float(np.dot(expired, salvage_values))
        return expired, salvage_revenue

    def _realized_demand(self, expected_demand: np.ndarray) -> np.ndarray:
        if not self.stochastic:
            return expected_demand.copy()
        noise_width = self.matrices["demand_params"][:, 2]
        shocks = self.rng.uniform(-noise_width, noise_width)
        return np.maximum(0.0, expected_demand + shocks)

    def _demand_params_at_day(self, day: int) -> tuple[np.ndarray, np.ndarray]:
        params = self.matrices["demand_params"]
        intercepts = params[:, 0] * np.exp(-params[:, 3] * day)
        slopes = params[:, 1] * (1.0 + params[:, 4] * day)
        return intercepts.astype(float), slopes.astype(float)

    def _sku_costs(self) -> np.ndarray:
        return self.matrices["costs"][: self.n_skus].astype(float)
