"""Closed-form oracle and structural recovery metrics for AVAL."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from catalog import ScenarioSpec, get_scenario


def demand_parameters_at_day(
    scenario: ScenarioSpec | str, day: int = 0, include_bundles: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    if day < 0:
        raise ValueError("day cannot be negative.")
    scenario_spec = get_scenario(scenario)
    params = scenario_spec.to_matrices(include_bundles=include_bundles)[
        "demand_params"
    ]
    intercepts = params[:, 0] * np.exp(-params[:, 3] * day)
    slopes = params[:, 1] * (1.0 + params[:, 4] * day)
    return intercepts.astype(float), slopes.astype(float)


def monopoly_price(intercept: float, slope: float, cost: float) -> float:
    """Closed-form monopoly price for D(p) = a - bp.

    p* = (a + bc) / (2b)
    """

    if intercept <= 0.0:
        raise ValueError("intercept must be positive.")
    if slope <= 0.0:
        raise ValueError("slope must be positive.")
    if cost < 0.0:
        raise ValueError("cost cannot be negative.")
    return float((intercept + slope * cost) / (2.0 * slope))


def monopoly_prices(
    intercepts: Sequence[float] | np.ndarray,
    slopes: Sequence[float] | np.ndarray,
    costs: Sequence[float] | np.ndarray,
) -> np.ndarray:
    intercepts_array = np.asarray(intercepts, dtype=float)
    slopes_array = np.asarray(slopes, dtype=float)
    costs_array = np.asarray(costs, dtype=float)
    if not (
        intercepts_array.shape == slopes_array.shape == costs_array.shape
    ):
        raise ValueError("intercepts, slopes, and costs must have the same shape.")
    if np.any(intercepts_array <= 0.0):
        raise ValueError("all intercepts must be positive.")
    if np.any(slopes_array <= 0.0):
        raise ValueError("all slopes must be positive.")
    if np.any(costs_array < 0.0):
        raise ValueError("costs cannot be negative.")
    return (intercepts_array + slopes_array * costs_array) / (2.0 * slopes_array)


def linear_demand(
    intercepts: Sequence[float] | np.ndarray,
    slopes: Sequence[float] | np.ndarray,
    prices: Sequence[float] | np.ndarray,
) -> np.ndarray:
    intercepts_array = np.asarray(intercepts, dtype=float)
    slopes_array = np.asarray(slopes, dtype=float)
    prices_array = np.asarray(prices, dtype=float)
    if not (
        intercepts_array.shape == slopes_array.shape == prices_array.shape
    ):
        raise ValueError("intercepts, slopes, and prices must have the same shape.")
    return np.maximum(0.0, intercepts_array - slopes_array * prices_array)


def oracle_price_vector(
    scenario: ScenarioSpec | str, day: int = 0, include_bundles: bool = True
) -> np.ndarray:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=include_bundles)
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=include_bundles
    )
    return monopoly_prices(intercepts, slopes, matrices["costs"])


def oracle_monopoly_outcome(
    scenario: ScenarioSpec | str, day: int = 0, include_bundles: bool = True
) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=include_bundles)
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=include_bundles
    )
    prices = monopoly_prices(intercepts, slopes, matrices["costs"])
    quantities = linear_demand(intercepts, slopes, prices)
    profits = (prices - matrices["costs"]) * quantities
    return {
        "scenario_id": scenario_spec.scenario_id,
        "day": day,
        "offer_ids": matrices["offer_ids"].copy(),
        "prices": prices,
        "quantities": quantities,
        "profits": profits,
        "total_profit": float(profits.sum()),
        "intercepts": intercepts,
        "slopes": slopes,
        "costs": matrices["costs"].copy(),
    }


def newsvendor_critical_fractile(
    price: float, cost: float, salvage_value: float = 0.0
) -> float:
    if price <= 0.0:
        raise ValueError("price must be positive.")
    if cost < 0.0:
        raise ValueError("cost cannot be negative.")
    if salvage_value < 0.0:
        raise ValueError("salvage_value cannot be negative.")
    denominator = price - salvage_value
    if denominator <= 0.0 or price <= cost:
        return 0.0
    return float(np.clip((price - cost) / denominator, 0.0, 1.0))


def clipped_uniform_quantile(
    center: float, half_width: float, probability: float
) -> float:
    """Quantile of max(0, center + U[-half_width, half_width])."""

    if half_width < 0.0:
        raise ValueError("half_width cannot be negative.")
    if probability < 0.0 or probability > 1.0:
        raise ValueError("probability must be in [0, 1].")
    if probability <= 0.0:
        return 0.0
    if half_width == 0.0:
        return max(0.0, float(center))

    lower = center - half_width
    upper = center + half_width
    if upper <= 0.0:
        return 0.0
    mass_at_zero = float(np.clip((0.0 - lower) / (2.0 * half_width), 0.0, 1.0))
    if probability <= mass_at_zero:
        return 0.0
    return float(max(0.0, lower + 2.0 * half_width * probability))


def newsvendor_quantity(
    intercept: float,
    slope: float,
    price: float,
    cost: float,
    salvage_value: float,
    noise_width: float,
) -> float:
    if slope <= 0.0:
        raise ValueError("slope must be positive.")
    critical_fractile = newsvendor_critical_fractile(price, cost, salvage_value)
    if critical_fractile <= 0.0:
        return 0.0
    demand_center = intercept - slope * price
    return clipped_uniform_quantile(demand_center, noise_width, critical_fractile)


def newsvendor_quantities(
    scenario: ScenarioSpec | str,
    prices: Sequence[float] | np.ndarray | None = None,
    day: int = 0,
) -> np.ndarray:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=False)
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=False
    )
    prices_array = (
        oracle_price_vector(scenario_spec, day=day, include_bundles=False)
        if prices is None
        else np.asarray(prices, dtype=float)
    )
    if prices_array.shape != (len(scenario_spec.skus),):
        raise ValueError(
            f"Expected {len(scenario_spec.skus)} SKU prices, received {prices_array.size}."
        )
    return np.array(
        [
            newsvendor_quantity(
                intercept=float(intercepts[idx]),
                slope=float(slopes[idx]),
                price=float(prices_array[idx]),
                cost=float(matrices["costs"][idx]),
                salvage_value=float(matrices["salvage_values"][idx]),
                noise_width=float(matrices["demand_params"][idx, 2]),
            )
            for idx in range(len(scenario_spec.skus))
        ],
        dtype=float,
    )


def oracle_policy(scenario: ScenarioSpec | str, day: int = 0) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=True)
    outcome = oracle_monopoly_outcome(scenario_spec, day=day, include_bundles=True)
    sku_prices = outcome["prices"][: len(scenario_spec.skus)]
    orders = newsvendor_quantities(scenario_spec, prices=sku_prices, day=day)
    return {
        **outcome,
        "orders": orders,
        "sku_ids": tuple(sku.sku_id for sku in scenario_spec.skus),
        "bundle_component_matrix": matrices["bundle_component_matrix"].copy(),
    }


def cost_pass_through(
    costs: Sequence[float] | np.ndarray, prices: Sequence[float] | np.ndarray
) -> float:
    """Recover dp/dc as the OLS slope of prices on costs."""

    costs_array = np.asarray(costs, dtype=float)
    prices_array = np.asarray(prices, dtype=float)
    if costs_array.shape != prices_array.shape:
        raise ValueError("costs and prices must have the same shape.")
    if costs_array.size < 2:
        return float("nan")
    finite = np.isfinite(costs_array) & np.isfinite(prices_array)
    if finite.sum() < 2:
        return float("nan")
    centered_costs = costs_array[finite] - costs_array[finite].mean()
    centered_prices = prices_array[finite] - prices_array[finite].mean()
    denominator = float(np.dot(centered_costs, centered_costs))
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(centered_costs, centered_prices) / denominator)


def implied_lerner_elasticity(
    prices: Sequence[float] | np.ndarray, costs: Sequence[float] | np.ndarray
) -> np.ndarray:
    """Elasticity implied by the Lerner condition: (p-c)/p = -1/e."""

    prices_array = np.asarray(prices, dtype=float)
    costs_array = np.asarray(costs, dtype=float)
    if prices_array.shape != costs_array.shape:
        raise ValueError("prices and costs must have the same shape.")
    margin = prices_array - costs_array
    elasticities = np.full(prices_array.shape, np.nan, dtype=float)
    valid = (prices_array > 0.0) & (margin > 0.0)
    elasticities[valid] = -prices_array[valid] / margin[valid]
    return elasticities


def linear_demand_elasticity(
    intercepts: Sequence[float] | np.ndarray,
    slopes: Sequence[float] | np.ndarray,
    prices: Sequence[float] | np.ndarray,
) -> np.ndarray:
    intercepts_array = np.asarray(intercepts, dtype=float)
    slopes_array = np.asarray(slopes, dtype=float)
    prices_array = np.asarray(prices, dtype=float)
    quantities = linear_demand(intercepts_array, slopes_array, prices_array)
    elasticities = np.full(prices_array.shape, np.nan, dtype=float)
    valid = quantities > 0.0
    elasticities[valid] = -(slopes_array[valid] * prices_array[valid]) / quantities[
        valid
    ]
    return elasticities


def structural_recovery(
    scenario: ScenarioSpec | str,
    prices: Sequence[float] | np.ndarray,
    day: int = 0,
    include_bundles: bool = True,
) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=include_bundles)
    prices_array = np.asarray(prices, dtype=float)
    if prices_array.shape != matrices["costs"].shape:
        raise ValueError(
            f"Expected {matrices['costs'].size} prices, received {prices_array.size}."
        )
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=include_bundles
    )
    oracle_prices = monopoly_prices(intercepts, slopes, matrices["costs"])
    return {
        "scenario_id": scenario_spec.scenario_id,
        "day": day,
        "offer_ids": matrices["offer_ids"].copy(),
        "cost_pass_through": cost_pass_through(matrices["costs"], prices_array),
        "oracle_cost_pass_through": np.full(prices_array.shape, 0.5, dtype=float),
        "implied_lerner_elasticity": implied_lerner_elasticity(
            prices_array, matrices["costs"]
        ),
        "linear_demand_elasticity": linear_demand_elasticity(
            intercepts, slopes, prices_array
        ),
        "oracle_prices": oracle_prices,
        "absolute_price_gap": prices_array - oracle_prices,
        "relative_price_gap": np.divide(
            prices_array - oracle_prices,
            oracle_prices,
            out=np.zeros_like(prices_array),
            where=oracle_prices != 0.0,
        ),
    }


def r_squared(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Coefficient of determination for the OLS fit y ~ a + b x."""

    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    finite = np.isfinite(x_array) & np.isfinite(y_array)
    if finite.sum() < 2:
        return 0.0
    x_finite = x_array[finite]
    y_finite = y_array[finite]
    var_x = float(np.dot(x_finite - x_finite.mean(), x_finite - x_finite.mean()))
    var_y = float(np.dot(y_finite - y_finite.mean(), y_finite - y_finite.mean()))
    if var_x <= 0.0 or var_y <= 0.0:
        return 0.0
    covariance = float(
        np.dot(x_finite - x_finite.mean(), y_finite - y_finite.mean())
    )
    return float(np.clip(covariance * covariance / (var_x * var_y), 0.0, 1.0))


def elasticity_recovery(
    scenario: ScenarioSpec | str,
    prices: Sequence[float] | np.ndarray,
    day: int = 0,
    include_bundles: bool = True,
) -> dict[str, Any]:
    """Recover whether the agent's revealed elasticity tracks the true elasticity.

    The Lerner condition makes a rational monopolist's *implied* elasticity
    -p / (p - c) coincide with the *true* demand elasticity -b p / (a - b p) at
    its chosen price. Regressing implied on true across offers gives a slope and
    an R^2: a rational agent lands on slope ~ 1, R^2 ~ 1; a cost-plus agent holds
    a constant implied elasticity regardless of the truth, so its R^2 collapses.
    This is identified from a single price vector, unlike the cross-sectional
    cost pass-through, which conflates the per-product derivative with demand
    heterogeneity.
    """

    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=include_bundles)
    prices_array = np.asarray(prices, dtype=float)
    if prices_array.shape != matrices["costs"].shape:
        raise ValueError(
            f"Expected {matrices['costs'].size} prices, received {prices_array.size}."
        )
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=include_bundles
    )
    implied = implied_lerner_elasticity(prices_array, matrices["costs"])
    true = linear_demand_elasticity(intercepts, slopes, prices_array)
    finite = np.isfinite(implied) & np.isfinite(true)
    # When fewer than two offers are identifiable (e.g. demand has decayed below
    # cost so the optimal quantity is ~0, or the agent priced below cost so the
    # Lerner elasticity is undefined) there is no elasticity signal to score.
    # Return NaN so the step is excluded from the mean rather than scored 0.
    fit_r2 = r_squared(true, implied) if finite.sum() >= 2 else float("nan")
    if finite.sum() >= 2:
        true_finite = true[finite]
        implied_finite = implied[finite]
        var_true = float(
            np.dot(true_finite - true_finite.mean(), true_finite - true_finite.mean())
        )
        slope = (
            float(
                np.dot(
                    true_finite - true_finite.mean(),
                    implied_finite - implied_finite.mean(),
                )
                / var_true
            )
            if var_true > 0.0
            else float("nan")
        )
        gap = float(np.nanmean(np.abs(implied_finite - true_finite)))
    else:
        slope = float("nan")
        gap = float("nan")
    return {
        "scenario_id": scenario_spec.scenario_id,
        "day": day,
        "implied_elasticity": implied,
        "true_elasticity": true,
        "tracking_slope": slope,
        "tracking_r_squared": fit_r2,
        "mean_abs_elasticity_gap": gap,
        "n_identified": int(finite.sum()),
    }


def price_efficiency(
    scenario: ScenarioSpec | str,
    prices: Sequence[float] | np.ndarray,
    day: int = 0,
    include_bundles: bool = True,
) -> dict[str, Any]:
    scenario_spec = get_scenario(scenario)
    matrices = scenario_spec.to_matrices(include_bundles=include_bundles)
    prices_array = np.asarray(prices, dtype=float)
    if prices_array.shape != matrices["costs"].shape:
        raise ValueError(
            f"Expected {matrices['costs'].size} prices, received {prices_array.size}."
        )
    intercepts, slopes = demand_parameters_at_day(
        scenario_spec, day=day, include_bundles=include_bundles
    )
    oracle_prices = monopoly_prices(intercepts, slopes, matrices["costs"])
    oracle_profits = (oracle_prices - matrices["costs"]) * linear_demand(
        intercepts, slopes, oracle_prices
    )
    agent_profits = (prices_array - matrices["costs"]) * linear_demand(
        intercepts, slopes, prices_array
    )
    total_oracle_profit = float(oracle_profits.sum())
    total_agent_profit = float(agent_profits.sum())
    ratio = (
        total_agent_profit / total_oracle_profit
        if total_oracle_profit > 0.0
        else float("nan")
    )
    return {
        "scenario_id": scenario_spec.scenario_id,
        "day": day,
        "offer_ids": matrices["offer_ids"].copy(),
        "oracle_profit": total_oracle_profit,
        "agent_profit": total_agent_profit,
        "efficiency_ratio": ratio,
        "profit_gap": total_oracle_profit - total_agent_profit,
        "oracle_prices": oracle_prices,
    }

