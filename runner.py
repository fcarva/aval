"""Simulation runner for AVAL scenario batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from catalog import SCENARIOS, ScenarioSpec, get_scenario
from env import RetailEnv
from metrics import (
    elasticity_recovery,
    oracle_policy,
    price_efficiency,
    structural_recovery,
)
from scoring import aggregate_certificate, aval_certificate


class AgentLike(Protocol):
    def act(self, obs: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return an environment action for an observation."""


@dataclass(frozen=True)
class OracleAgent:
    """Deterministic reference agent backed by the closed-form oracle."""

    order_for_bundle_demand: bool = True

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        scenario_id = str(obs["scenario_id"])
        day = int(obs["day"])
        policy = oracle_policy(scenario_id, day=day)
        offer_ids = tuple(str(item) for item in obs["offer_ids"])
        sku_ids = tuple(str(item) for item in obs["sku_ids"])
        n_skus = len(sku_ids)

        target_inventory = np.asarray(policy["orders"], dtype=float)
        if self.order_for_bundle_demand:
            bundle_quantities = np.asarray(policy["quantities"][n_skus:], dtype=float)
            component_matrix = np.asarray(policy["bundle_component_matrix"], dtype=float)
            if bundle_quantities.size:
                target_inventory = target_inventory + component_matrix.T @ bundle_quantities

        current_inventory = np.asarray(obs["inventory"], dtype=float)
        orders = np.maximum(0.0, target_inventory - current_inventory)
        return {
            "prices": {
                offer_id: float(price)
                for offer_id, price in zip(offer_ids, policy["prices"])
            },
            "orders": {
                sku_id: float(order)
                for sku_id, order in zip(sku_ids, orders)
            },
            "markdowns": {offer_id: 0.0 for offer_id in offer_ids},
        }


@dataclass(frozen=True)
class NaiveAgent:
    """Intentionally weak retail policy used to demonstrate AVAL diagnostics."""

    order_multiplier: float = 1.60
    bundle_price_multiplier: float = 1.35

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        offer_ids = tuple(str(item) for item in obs["offer_ids"])
        sku_ids = tuple(str(item) for item in obs["sku_ids"])
        bundle_ids = set(str(item) for item in obs.get("bundle_ids", ()))
        prices = np.asarray(obs["current_prices"], dtype=float)
        costs = np.asarray(obs["costs"], dtype=float)
        expected_demand = np.asarray(obs["expected_demand"], dtype=float)

        price_action = {
            offer_id: float(
                costs[idx] * self.bundle_price_multiplier
                if offer_id in bundle_ids
                else prices[idx]
            )
            for idx, offer_id in enumerate(offer_ids)
        }
        orders = {
            sku_id: float(max(0.0, expected_demand[idx] * self.order_multiplier))
            for idx, sku_id in enumerate(sku_ids)
        }
        return {
            "prices": price_action,
            "orders": orders,
            "markdowns": {offer_id: 0.0 for offer_id in offer_ids},
        }


@dataclass(frozen=True)
class FixedMarkupAgent:
    """Naive human baseline: fixed cost-plus markup, demand-anchored restock.

    Reproduces the doc's FixedMarkup archetype. Pricing ignores elasticity (so it
    leaves money on the table and its implied elasticity is flat), but ordering
    tracks expected demand closely enough to survive.
    """

    markup: float = 0.50
    restock_buffer: float = 1.10

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        offer_ids = tuple(str(item) for item in obs["offer_ids"])
        sku_ids = tuple(str(item) for item in obs["sku_ids"])
        costs = np.asarray(obs["costs"], dtype=float)
        inventory = np.asarray(obs["inventory"], dtype=float)
        expected_demand = np.asarray(obs["expected_demand"], dtype=float)
        prices = {
            offer_id: float(max(0.01, costs[idx] * (1.0 + self.markup)))
            for idx, offer_id in enumerate(offer_ids)
        }
        orders = {
            sku_id: float(
                max(0.0, expected_demand[idx] * self.restock_buffer - inventory[idx])
            )
            for idx, sku_id in enumerate(sku_ids)
        }
        return {
            "prices": prices,
            "orders": orders,
            "markdowns": {offer_id: 0.0 for offer_id in offer_ids},
        }


class RandomAgent:
    """Erratic baseline: random multipliers on cost, random restock.

    Demonstrates that volatility is fatal. Some prices land below cost (a
    dominated action), pricing is incoherent across days, and over-ordering can
    drive the account negative, so survival is well below 100%.
    """

    def __init__(
        self,
        seed: int = 0,
        price_low: float = 0.6,
        price_high: float = 3.5,
        order_high: float = 30.0,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.price_low = float(price_low)
        self.price_high = float(price_high)
        self.order_high = float(order_high)

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        offer_ids = tuple(str(item) for item in obs["offer_ids"])
        sku_ids = tuple(str(item) for item in obs["sku_ids"])
        costs = np.asarray(obs["costs"], dtype=float)
        prices = {
            offer_id: float(
                max(0.01, costs[idx] * self.rng.uniform(self.price_low, self.price_high))
            )
            for idx, offer_id in enumerate(offer_ids)
        }
        orders = {
            sku_id: float(self.rng.uniform(0.0, self.order_high)) for sku_id in sku_ids
        }
        return {
            "prices": prices,
            "orders": orders,
            "markdowns": {offer_id: 0.0 for offer_id in offer_ids},
        }


def run_episode(
    agent: AgentLike | Callable[[Mapping[str, Any]], Mapping[str, Any]],
    scenario: ScenarioSpec | str,
    seed: int,
    stochastic: bool = True,
    max_days: int | None = None,
) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    env = RetailEnv(scenario_spec, seed=seed, stochastic=stochastic)
    obs = env.reset()
    records: list[dict[str, Any]] = []
    done = False

    while not done:
        day = int(obs["day"])
        if max_days is not None and day >= max_days:
            break
        action = _call_agent(agent, obs)
        step = env.step(action)
        record = _build_step_record(
            scenario_spec=scenario_spec,
            seed=seed,
            day=day,
            obs=obs,
            action=action,
            info=step.info,
        )
        records.append(record)
        obs = step.observation
        done = step.done

    summary = summarize_episode(scenario_spec, seed, records, obs)
    return {
        "scenario_id": scenario_spec.scenario_id,
        "seed": int(seed),
        "horizon_days": scenario_spec.horizon_days,
        "steps": records,
        "summary": summary,
    }


def run_multi_seed(
    agent: AgentLike | Callable[[Mapping[str, Any]], Mapping[str, Any]],
    seeds: Sequence[int],
    scenario_ids: Sequence[str] | None = None,
    stochastic: bool = True,
    scenario_rng_seed: int = 0,
    report_path: str | None = None,
) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    scenario_assignment = sample_scenarios_for_seeds(
        seeds=seeds,
        scenario_ids=scenario_ids,
        rng_seed=scenario_rng_seed,
    )
    episodes = [
        run_episode(
            agent=agent,
            scenario=scenario_assignment[int(seed)],
            seed=int(seed),
            stochastic=stochastic,
        )
        for seed in seeds
    ]
    batch = {
        "agent_name": agent.__class__.__name__,
        "seeds": [int(seed) for seed in seeds],
        "scenario_assignment": scenario_assignment,
        "episodes": episodes,
        "summary": summarize_batch(episodes),
    }
    if report_path is not None:
        from report import write_html_report

        batch["report_path"] = write_html_report(batch, report_path)
    return batch


def sample_scenarios_for_seeds(
    seeds: Sequence[int],
    scenario_ids: Sequence[str] | None = None,
    rng_seed: int = 0,
) -> dict[int, str]:
    if not seeds:
        raise ValueError("seeds cannot be empty.")
    ids = tuple(scenario_ids or SCENARIOS.keys())
    if not ids:
        raise ValueError("scenario_ids cannot be empty.")
    for scenario_id in ids:
        get_scenario(scenario_id)

    rng = np.random.default_rng(rng_seed)
    repeats = int(np.ceil(len(seeds) / len(ids)))
    pool = list(ids) * repeats
    rng.shuffle(pool)
    return {int(seed): str(pool[idx]) for idx, seed in enumerate(seeds)}


def summarize_episode(
    scenario: ScenarioSpec | str,
    seed: int,
    records: Sequence[Mapping[str, Any]],
    final_obs: Mapping[str, Any],
) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    if not records:
        raise ValueError("records cannot be empty.")

    total_profit = _sum(records, "profit")
    total_revenue = _sum(records, "revenue")
    total_order_cost = _sum(records, "order_cost")
    total_expired_units = _sum_vector(records, "expired_units")
    total_lost_sales = _sum_vector(records, "lost_sales")
    total_orders = _sum_vector(records, "orders")
    total_bundle_sales = _sum_vector(records, "bundle_sales")
    final_inventory = _float_list(final_obs["inventory"])
    mean_price_efficiency = _mean(records, "price_efficiency")
    mean_abs_relative_price_gap = _mean(records, "mean_abs_relative_price_gap")
    mean_abs_order_gap = _mean(records, "mean_abs_order_gap")
    mean_late_markdown = _late_markdown_mean(scenario_spec, records)
    mean_late_relative_price_gap = _late_price_gap_mean(scenario_spec, records)
    mean_bundle_relative_price_gap = _bundle_price_gap_mean(scenario_spec, records)
    mean_structural_r_squared = _mean(records, "structural_r_squared")
    mean_order_accuracy = _order_accuracy(records)
    survived, min_cash = _survival(records)
    flag_rate, below_cost_total, over_capacity_total = _flag_rate(records)
    price_incoherence = _price_incoherence(scenario_spec, records)

    summary = {
        "scenario_id": scenario_spec.scenario_id,
        "seed": int(seed),
        "days_run": len(records),
        "total_profit": total_profit,
        "total_revenue": total_revenue,
        "total_order_cost": total_order_cost,
        "total_expired_units": total_expired_units,
        "total_lost_sales": total_lost_sales,
        "total_orders": total_orders,
        "total_bundle_sales": total_bundle_sales,
        "final_inventory": final_inventory,
        "mean_price_efficiency": mean_price_efficiency,
        "mean_abs_relative_price_gap": mean_abs_relative_price_gap,
        "mean_abs_order_gap": mean_abs_order_gap,
        "mean_order_accuracy": mean_order_accuracy,
        "mean_structural_r_squared": mean_structural_r_squared,
        "mean_late_markdown": mean_late_markdown,
        "mean_late_relative_price_gap": mean_late_relative_price_gap,
        "mean_bundle_relative_price_gap": mean_bundle_relative_price_gap,
        "survived": survived,
        "min_cash": min_cash,
        "flag_rate": flag_rate,
        "below_cost_actions": below_cost_total,
        "over_capacity_actions": over_capacity_total,
        "price_incoherence": price_incoherence,
    }
    summary["diagnostic"] = build_structural_diagnostic(summary)
    summary["certificate"] = aval_certificate(summary)
    summary["aval_score"] = summary["certificate"]["aval_score"]
    summary["grade"] = summary["certificate"]["grade"]
    summary["verdict"] = summary["certificate"]["verdict"]
    return summary


def summarize_batch(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("episodes cannot be empty.")
    scenario_ids = sorted({str(episode["scenario_id"]) for episode in episodes})
    by_scenario: dict[str, dict[str, Any]] = {}
    for scenario_id in scenario_ids:
        scenario_episodes = [
            episode for episode in episodes if episode["scenario_id"] == scenario_id
        ]
        summaries = [episode["summary"] for episode in scenario_episodes]
        by_scenario[scenario_id] = {
            "episodes": len(scenario_episodes),
            "mean_price_efficiency": _mean(summaries, "mean_price_efficiency"),
            "mean_abs_relative_price_gap": _mean(
                summaries, "mean_abs_relative_price_gap"
            ),
            "mean_abs_order_gap": _mean(summaries, "mean_abs_order_gap"),
            "mean_aval_score": _mean(summaries, "aval_score"),
            "survival_rate": _survival_rate(summaries),
            "total_profit": _sum(summaries, "total_profit"),
            "diagnostics": [summary["diagnostic"] for summary in summaries],
        }

    all_summaries = [episode["summary"] for episode in episodes]
    scores = [float(summary.get("aval_score", 0.0)) for summary in all_summaries]
    certificate = aggregate_certificate(scores)
    return {
        "episodes": len(episodes),
        "scenarios": by_scenario,
        "mean_price_efficiency": _mean(all_summaries, "mean_price_efficiency"),
        "mean_abs_relative_price_gap": _mean(
            all_summaries, "mean_abs_relative_price_gap"
        ),
        "mean_abs_order_gap": _mean(all_summaries, "mean_abs_order_gap"),
        "mean_structural_r_squared": _mean(all_summaries, "mean_structural_r_squared"),
        "mean_aval_score": certificate["aval_score"],
        "grade": certificate["grade"],
        "verdict": certificate["verdict"],
        "survival_rate": _survival_rate(all_summaries),
        "total_profit": _sum(all_summaries, "total_profit"),
    }


def build_structural_diagnostic(summary: Mapping[str, Any]) -> dict[str, Any]:
    scenario_id = str(summary["scenario_id"])
    if scenario_id == "buybye_autonomous":
        n_skus = len(get_scenario(scenario_id).skus)
        expired = float(sum(summary.get("total_expired_units", [])))
        orders = float(sum(summary.get("total_orders", [])))
        lost = float(sum(summary.get("total_lost_sales", [])[:n_skus]))
        expired_rate = expired / orders if orders > 0.0 else 0.0
        lost_rate = lost / (lost + orders) if lost + orders > 0.0 else 0.0
        failed = expired_rate > 0.15 or lost_rate > 0.15
        message = (
            "Falha principal: gerir estoque de pereciveis; perdas e rupturas "
            "ficaram distantes do Q* Newsvendor."
            if failed
            else "Gerir estoque perecivel ficou coerente com o Q* Newsvendor."
        )
        return {
            "scenario_id": scenario_id,
            "status": "fail" if failed else "pass",
            "failure_mode": "gerir estoque" if failed else "gerir estoque eficiente",
            "message": message,
            "signals": {
                "expired_rate": expired_rate,
                "lost_rate": lost_rate,
                "mean_abs_order_gap": float(summary.get("mean_abs_order_gap", 0.0)),
            },
        }

    if scenario_id == "d2c_fashion":
        final_inventory = float(sum(summary.get("final_inventory", [])))
        late_markdown = float(summary.get("mean_late_markdown", 0.0))
        late_price_gap = float(summary.get("mean_late_relative_price_gap", 0.0))
        # Outcome-based: leftover clearance stock while the effective price stayed
        # well above the oracle. Repricing and markdowns are both valid levers, so
        # we judge the price gap, not whether a specific lever was used.
        failed = final_inventory > 1.0 and late_price_gap > 0.10
        message = (
            "Falha principal: liquidar moda no ciclo D2C; markdowns e precos "
            "ficaram lentos frente a queda do hype."
            if failed
            else "Liquidar moda D2C acompanha a queda do hype."
        )
        return {
            "scenario_id": scenario_id,
            "status": "fail" if failed else "pass",
            "failure_mode": "liquidar moda" if failed else "liquidar moda eficiente",
            "message": message,
            "signals": {
                "final_inventory": final_inventory,
                "mean_late_markdown": late_markdown,
                "mean_late_relative_price_gap": late_price_gap,
            },
        }

    if scenario_id == "baco_premium":
        bundle_gap = abs(float(summary.get("mean_bundle_relative_price_gap", 0.0)))
        bundle_sales = float(sum(summary.get("total_bundle_sales", [])))
        failed = bundle_gap > 0.10
        message = (
            "Falha principal: precificar pacotes premium; bundles desviaram "
            "do preco otimo de monopolio."
            if failed
            else "Precificar pacotes premium ficou proximo do preco otimo de monopolio."
        )
        return {
            "scenario_id": scenario_id,
            "status": "fail" if failed else "pass",
            "failure_mode": "precificar pacotes" if failed else "precificar pacotes eficiente",
            "message": message,
            "signals": {
                "mean_bundle_relative_price_gap": bundle_gap,
                "total_bundle_sales": bundle_sales,
            },
        }

    return {
        "scenario_id": scenario_id,
        "status": "unknown",
        "failure_mode": "nao classificado",
        "message": "Cenario sem regra estrutural de diagnostico.",
        "signals": {},
    }


def _build_step_record(
    scenario_spec: ScenarioSpec,
    seed: int,
    day: int,
    obs: Mapping[str, Any],
    action: Mapping[str, Any],
    info: Mapping[str, Any],
) -> dict[str, Any]:
    prices = np.asarray(info["prices"], dtype=float)
    orders = np.asarray(info["orders"], dtype=float)
    orders_requested = np.asarray(info["orders_requested"], dtype=float)
    costs = np.asarray(obs["costs"], dtype=float)
    efficiency = price_efficiency(scenario_spec, prices, day=day)
    recovery = structural_recovery(scenario_spec, prices, day=day)
    elasticity = elasticity_recovery(scenario_spec, prices, day=day)
    policy = oracle_policy(scenario_spec, day=day)
    oracle_orders = np.asarray(policy["orders"], dtype=float)
    markdowns = _action_markdowns(action, obs)

    relative_gap = np.asarray(recovery["relative_price_gap"], dtype=float)
    order_gap = orders - oracle_orders
    below_cost_count = int(np.sum(prices < costs - 1e-9))
    over_capacity_count = int(np.sum(orders_requested > orders + 1e-6))
    return {
        "scenario_id": scenario_spec.scenario_id,
        "seed": int(seed),
        "day": int(day),
        "prices": _float_list(prices),
        "orders": _float_list(orders),
        "markdowns": _float_list(markdowns),
        "offer_ids": [str(item) for item in obs["offer_ids"]],
        "sku_ids": [str(item) for item in obs["sku_ids"]],
        "bundle_ids": [str(item) for item in obs.get("bundle_ids", ())],
        "oracle_prices": _float_list(efficiency["oracle_prices"]),
        "oracle_orders": _float_list(oracle_orders),
        "relative_price_gap": _float_list(relative_gap),
        "order_gap": _float_list(order_gap),
        "mean_abs_relative_price_gap": _nanmean_abs(relative_gap),
        "mean_abs_order_gap": _nanmean_abs(order_gap),
        "price_efficiency": float(efficiency["efficiency_ratio"]),
        "cost_pass_through": float(recovery["cost_pass_through"]),
        "structural_r_squared": float(elasticity["tracking_r_squared"]),
        "elasticity_tracking_slope": float(elasticity["tracking_slope"]),
        "below_cost_count": below_cost_count,
        "over_capacity_count": over_capacity_count,
        "abs_order_gap_sum": float(np.nansum(np.abs(order_gap))),
        "oracle_orders_sum": float(np.nansum(np.abs(oracle_orders))),
        "n_offers": len(obs["offer_ids"]),
        "n_skus": len(obs["sku_ids"]),
        "cash": (None if info.get("cash") is None else float(info["cash"])),
        "bankrupt": bool(info.get("bankrupt", False)),
        "profit": float(info["profit"]),
        "revenue": float(info["revenue"]),
        "order_cost": float(info["order_cost"]),
        "expired_units": _float_list(info["expired_units"]),
        "lost_sales": _float_list(info["lost_sales"]),
        "bundle_sales": _float_list(info["bundle_sales"]),
        "inventory_end": _float_list(info["inventory_end"]),
    }


def _call_agent(
    agent: AgentLike | Callable[[Mapping[str, Any]], Mapping[str, Any]],
    obs: Mapping[str, Any],
) -> Mapping[str, Any]:
    act = getattr(agent, "act", None)
    if callable(act):
        action = act(obs)
    elif callable(agent):
        action = agent(obs)
    else:
        raise TypeError("agent must be callable or expose act(obs).")
    if not isinstance(action, Mapping):
        raise TypeError("agent action must be a mapping.")
    return action


def _action_markdowns(action: Mapping[str, Any], obs: Mapping[str, Any]) -> np.ndarray:
    offer_ids = tuple(str(item) for item in obs["offer_ids"])
    raw = action.get("markdowns")
    if raw is None:
        return np.zeros(len(offer_ids), dtype=float)
    if isinstance(raw, Mapping):
        return np.array([float(raw.get(offer_id, 0.0)) for offer_id in offer_ids])
    array = np.asarray(raw, dtype=float)
    if array.shape != (len(offer_ids),):
        return np.zeros(len(offer_ids), dtype=float)
    return array


def _late_markdown_mean(
    scenario: ScenarioSpec, records: Sequence[Mapping[str, Any]]
) -> float:
    final_clearance_day = scenario.metadata.get("final_clearance_day")
    if final_clearance_day is None:
        return 0.0
    late_records = [record for record in records if record["day"] >= final_clearance_day]
    if not late_records:
        return 0.0
    return _mean(late_records, "markdowns")


def _late_price_gap_mean(
    scenario: ScenarioSpec, records: Sequence[Mapping[str, Any]]
) -> float:
    final_clearance_day = scenario.metadata.get("final_clearance_day")
    if final_clearance_day is None:
        return 0.0
    late_records = [record for record in records if record["day"] >= final_clearance_day]
    if not late_records:
        return 0.0
    return _mean(late_records, "mean_abs_relative_price_gap")


def _bundle_price_gap_mean(
    scenario: ScenarioSpec, records: Sequence[Mapping[str, Any]]
) -> float:
    if not scenario.bundles:
        return 0.0
    n_skus = len(scenario.skus)
    gaps: list[float] = []
    for record in records:
        relative_gap = record.get("relative_price_gap", [])
        gaps.extend(float(value) for value in relative_gap[n_skus:])
    if not gaps:
        return 0.0
    return float(np.nanmean(gaps))


def _survival_rate(summaries: Sequence[Mapping[str, Any]]) -> float:
    if not summaries:
        return 0.0
    survived = sum(1 for summary in summaries if summary.get("survived", True))
    return float(survived / len(summaries))


def _order_accuracy(records: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of the Newsvendor order policy the agent reproduces, in [0, 1]."""
    gap = float(np.nansum([float(record.get("abs_order_gap_sum", 0.0)) for record in records]))
    scale = float(np.nansum([float(record.get("oracle_orders_sum", 0.0)) for record in records]))
    if scale <= 0.0:
        return 1.0 if gap <= 0.0 else 0.0
    return float(np.clip(1.0 - gap / scale, 0.0, 1.0))


def _survival(records: Sequence[Mapping[str, Any]]) -> tuple[bool, float | None]:
    cash_values = [record.get("cash") for record in records if record.get("cash") is not None]
    survived = not any(bool(record.get("bankrupt", False)) for record in records)
    min_cash = float(min(cash_values)) if cash_values else None
    return survived, min_cash


def _flag_rate(records: Sequence[Mapping[str, Any]]) -> tuple[float, int, int]:
    """Rationality-flag rate over all decision slots, plus raw flag counts."""
    below_cost = sum(int(record.get("below_cost_count", 0)) for record in records)
    over_capacity = sum(int(record.get("over_capacity_count", 0)) for record in records)
    slots = sum(
        int(record.get("n_offers", 0)) + int(record.get("n_skus", 0)) for record in records
    )
    if slots <= 0:
        return 0.0, below_cost, over_capacity
    return float(np.clip((below_cost + over_capacity) / slots, 0.0, 1.0)), below_cost, over_capacity


def _price_incoherence(
    scenario: ScenarioSpec, records: Sequence[Mapping[str, Any]]
) -> float:
    """Coefficient of variation of each SKU price across days, for stationary demand.

    Under stationary fundamentals the rational monopoly price is constant, so a
    high coefficient of variation reveals revealed-preference incoherence (a
    practical GARP proxy). Non-stationary scenarios (hype decay or slope growth)
    are exempt because the optimal price is supposed to move.
    """
    stationary = all(
        sku.demand.hype_decay == 0.0 and sku.demand.slope_growth_per_day == 0.0
        for sku in scenario.skus
    )
    if not stationary or len(records) < 2:
        return 0.0
    n_skus = len(scenario.skus)
    price_matrix = np.array(
        [np.asarray(record.get("prices", []), dtype=float)[:n_skus] for record in records],
        dtype=float,
    )
    if price_matrix.size == 0:
        return 0.0
    means = price_matrix.mean(axis=0)
    stds = price_matrix.std(axis=0)
    valid = means > 0.0
    if not np.any(valid):
        return 0.0
    return float(np.clip(np.mean(stds[valid] / means[valid]), 0.0, 1.0))


def _sum(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(record.get(key, 0.0)) for record in records]
    return float(np.nansum(values))


def _sum_vector(records: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    arrays = [np.asarray(record.get(key, []), dtype=float) for record in records]
    if not arrays:
        return []
    max_len = max(array.size for array in arrays)
    total = np.zeros(max_len, dtype=float)
    for array in arrays:
        if array.size:
            total[: array.size] += array
    return _float_list(total)


def _mean(records: Sequence[Mapping[str, Any]], key: str) -> float:
    values: list[float] = []
    for record in records:
        raw = record.get(key)
        if raw is None:
            continue
        array = np.asarray(raw, dtype=float)
        if array.size == 0:
            continue
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            continue
        values.append(float(np.mean(finite)))
    if not values:
        return 0.0
    return float(np.nanmean(values))


def _nanmean_abs(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    return float(np.nanmean(np.abs(array)))


def _float_list(values: Any) -> list[float]:
    array = np.asarray(values, dtype=float)
    return [float(value) for value in array.reshape(-1)]


if __name__ == "__main__":
    batch = run_multi_seed(
        agent=NaiveAgent(),
        seeds=[101, 202, 303],
        stochastic=True,
        scenario_rng_seed=0,
        report_path="aval_parecer.html",
    )
    print(f"Wrote {batch['report_path']} with {len(batch['episodes'])} episodes.")
