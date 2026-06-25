"""Scenario catalog for the AVAL MVP.

The catalog is intentionally small and closed-form friendly. Every offer uses a
linear demand primitive, so the oracle can later recover monopoly prices from
the same (a, b, c) inputs without simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


DEMAND_PARAM_COLUMNS = (
    "intercept",
    "slope",
    "noise_width",
    "hype_decay",
    "slope_growth_per_day",
)


@dataclass(frozen=True)
class DemandSpec:
    """Linear demand D_t(p) = max(0, a_t - b_t * p).

    D2C scenarios use the same primitive with time-varying parameters:
    a_t = a * exp(-hype_decay * t)
    b_t = b * (1 + slope_growth_per_day * t)
    """

    intercept: float
    slope: float
    noise_width: float = 0.0
    hype_decay: float = 0.0
    slope_growth_per_day: float = 0.0

    def __post_init__(self) -> None:
        if self.intercept <= 0:
            raise ValueError("Demand intercept must be positive.")
        if self.slope <= 0:
            raise ValueError("Demand slope must be positive.")
        if self.noise_width < 0:
            raise ValueError("Demand noise width cannot be negative.")
        if self.hype_decay < 0:
            raise ValueError("Hype decay cannot be negative.")
        if self.slope_growth_per_day < 0:
            raise ValueError("Slope growth cannot be negative.")

    def params_at_day(self, day: int = 0) -> Tuple[float, float]:
        if day < 0:
            raise ValueError("day cannot be negative.")
        intercept_t = self.intercept * math.exp(-self.hype_decay * day)
        slope_t = self.slope * (1.0 + self.slope_growth_per_day * day)
        return intercept_t, slope_t

    def as_row(self) -> Tuple[float, float, float, float, float]:
        return (
            self.intercept,
            self.slope,
            self.noise_width,
            self.hype_decay,
            self.slope_growth_per_day,
        )


@dataclass(frozen=True)
class SKU:
    sku_id: str
    label: str
    cost: float
    initial_price: float
    shelf_life_days: int
    demand: DemandSpec
    salvage_value: float = 0.0
    max_order_units: int = 0

    def __post_init__(self) -> None:
        if not self.sku_id:
            raise ValueError("sku_id is required.")
        if self.cost <= 0:
            raise ValueError(f"{self.sku_id}: cost must be positive.")
        if self.initial_price <= 0:
            raise ValueError(f"{self.sku_id}: initial_price must be positive.")
        if self.shelf_life_days <= 0:
            raise ValueError(f"{self.sku_id}: shelf_life_days must be positive.")
        if self.salvage_value < 0:
            raise ValueError(f"{self.sku_id}: salvage_value cannot be negative.")
        if self.salvage_value >= self.cost:
            raise ValueError(f"{self.sku_id}: salvage_value must stay below cost.")
        if self.max_order_units < 0:
            raise ValueError(f"{self.sku_id}: max_order_units cannot be negative.")


@dataclass(frozen=True)
class BundleComponent:
    sku_id: str
    quantity: int = 1

    def __post_init__(self) -> None:
        if not self.sku_id:
            raise ValueError("Bundle component sku_id is required.")
        if self.quantity <= 0:
            raise ValueError("Bundle component quantity must be positive.")


@dataclass(frozen=True)
class BundleSpec:
    bundle_id: str
    label: str
    components: Tuple[BundleComponent, ...]
    initial_price: float
    demand: DemandSpec
    tier: str = "curated"

    def __post_init__(self) -> None:
        if not self.bundle_id:
            raise ValueError("bundle_id is required.")
        if not self.components:
            raise ValueError(f"{self.bundle_id}: components are required.")
        if self.initial_price <= 0:
            raise ValueError(f"{self.bundle_id}: initial_price must be positive.")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    label: str
    engine: str
    horizon_days: int
    skus: Tuple[SKU, ...]
    bundles: Tuple[BundleSpec, ...] = ()
    capacity_units: int | None = None
    starting_cash: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError("scenario_id is required.")
        if self.horizon_days <= 0:
            raise ValueError(f"{self.scenario_id}: horizon_days must be positive.")
        if not self.skus:
            raise ValueError(f"{self.scenario_id}: at least one SKU is required.")
        if self.capacity_units is not None and self.capacity_units <= 0:
            raise ValueError(f"{self.scenario_id}: capacity_units must be positive.")
        if self.starting_cash is not None and self.starting_cash <= 0:
            raise ValueError(f"{self.scenario_id}: starting_cash must be positive.")
        _assert_unique([sku.sku_id for sku in self.skus], "sku_id")
        _assert_unique([bundle.bundle_id for bundle in self.bundles], "bundle_id")
        sku_ids = {sku.sku_id for sku in self.skus}
        for bundle in self.bundles:
            for component in bundle.components:
                if component.sku_id not in sku_ids:
                    raise ValueError(
                        f"{bundle.bundle_id}: unknown component {component.sku_id}."
                    )

    def sku_index(self) -> Dict[str, int]:
        return {sku.sku_id: idx for idx, sku in enumerate(self.skus)}

    def bundle_cost(self, bundle: BundleSpec) -> float:
        sku_by_id = {sku.sku_id: sku for sku in self.skus}
        return float(
            sum(
                sku_by_id[component.sku_id].cost * component.quantity
                for component in bundle.components
            )
        )

    def bundle_shelf_life_days(self, bundle: BundleSpec) -> int:
        sku_by_id = {sku.sku_id: sku for sku in self.skus}
        return min(
            sku_by_id[component.sku_id].shelf_life_days
            for component in bundle.components
        )

    def offer_ids(self, include_bundles: bool = True) -> Tuple[str, ...]:
        ids = tuple(sku.sku_id for sku in self.skus)
        if include_bundles:
            ids = ids + tuple(bundle.bundle_id for bundle in self.bundles)
        return ids

    def to_matrices(self, include_bundles: bool = True) -> Dict[str, object]:
        """Return NumPy matrices used by env, oracle, and reporting layers."""

        offers: list[SKU | BundleSpec] = list(self.skus)
        if include_bundles:
            offers.extend(self.bundles)

        costs: list[float] = []
        prices: list[float] = []
        shelf_life: list[int] = []
        demand_rows: list[Tuple[float, float, float, float, float]] = []
        salvage_values: list[float] = []
        max_orders: list[int] = []
        is_bundle: list[bool] = []

        for offer in offers:
            if isinstance(offer, SKU):
                costs.append(offer.cost)
                prices.append(offer.initial_price)
                shelf_life.append(offer.shelf_life_days)
                demand_rows.append(offer.demand.as_row())
                salvage_values.append(offer.salvage_value)
                max_orders.append(offer.max_order_units)
                is_bundle.append(False)
            else:
                costs.append(self.bundle_cost(offer))
                prices.append(offer.initial_price)
                shelf_life.append(self.bundle_shelf_life_days(offer))
                demand_rows.append(offer.demand.as_row())
                salvage_values.append(0.0)
                max_orders.append(0)
                is_bundle.append(True)

        return {
            "scenario_id": self.scenario_id,
            "engine": self.engine,
            "offer_ids": np.array(self.offer_ids(include_bundles), dtype=object),
            "costs": np.array(costs, dtype=float),
            "initial_prices": np.array(prices, dtype=float),
            "shelf_life_days": np.array(shelf_life, dtype=int),
            "demand_params": np.array(demand_rows, dtype=float),
            "demand_param_columns": DEMAND_PARAM_COLUMNS,
            "salvage_values": np.array(salvage_values, dtype=float),
            "max_order_units": np.array(max_orders, dtype=int),
            "is_bundle": np.array(is_bundle, dtype=bool),
            "bundle_component_matrix": self.bundle_component_matrix(),
        }

    def bundle_component_matrix(self) -> np.ndarray:
        matrix = np.zeros((len(self.bundles), len(self.skus)), dtype=int)
        sku_index = self.sku_index()
        for row_idx, bundle in enumerate(self.bundles):
            for component in bundle.components:
                matrix[row_idx, sku_index[component.sku_id]] = component.quantity
        return matrix


def demand_at_price(
    demand: DemandSpec, price: float | np.ndarray, day: int = 0
) -> float | np.ndarray:
    """Evaluate expected linear demand at a price and simulation day."""

    intercept_t, slope_t = demand.params_at_day(day)
    raw = intercept_t - slope_t * np.asarray(price, dtype=float)
    result = np.maximum(0.0, raw)
    if np.ndim(result) == 0:
        return float(result)
    return result


def expected_demand_vector(
    scenario: ScenarioSpec | str,
    prices: Sequence[float],
    day: int = 0,
    include_bundles: bool = True,
) -> np.ndarray:
    scenario_spec = get_scenario(scenario) if isinstance(scenario, str) else scenario
    demand_specs: list[DemandSpec] = [sku.demand for sku in scenario_spec.skus]
    if include_bundles:
        demand_specs.extend(bundle.demand for bundle in scenario_spec.bundles)
    if len(prices) != len(demand_specs):
        raise ValueError(
            f"Expected {len(demand_specs)} prices, received {len(prices)}."
        )
    return np.array(
        [
            demand_at_price(demand, float(price), day=day)
            for demand, price in zip(demand_specs, prices)
        ],
        dtype=float,
    )


def get_scenario(scenario: ScenarioSpec | str) -> ScenarioSpec:
    if isinstance(scenario, ScenarioSpec):
        return scenario
    try:
        return SCENARIOS[scenario]
    except KeyError as exc:
        known = ", ".join(sorted(SCENARIOS))
        raise KeyError(f"Unknown scenario '{scenario}'. Known scenarios: {known}.") from exc


def validate_catalog(scenarios: Mapping[str, ScenarioSpec]) -> None:
    _assert_unique(list(scenarios), "scenario_id")
    for scenario_id, scenario in scenarios.items():
        if scenario_id != scenario.scenario_id:
            raise ValueError(f"Catalog key {scenario_id} does not match scenario id.")
        matrices = scenario.to_matrices(include_bundles=True)
        costs = matrices["costs"]
        prices = matrices["initial_prices"]
        demand_params = matrices["demand_params"]
        if not isinstance(costs, np.ndarray) or not np.all(costs > 0):
            raise ValueError(f"{scenario_id}: all costs must be positive.")
        if not isinstance(prices, np.ndarray) or not np.all(prices > 0):
            raise ValueError(f"{scenario_id}: all prices must be positive.")
        if not isinstance(demand_params, np.ndarray) or demand_params.shape[1] != 5:
            raise ValueError(f"{scenario_id}: invalid demand parameter matrix.")
        if not np.all(demand_params[:, 0] > 0):
            raise ValueError(f"{scenario_id}: demand intercepts must be positive.")
        if not np.all(demand_params[:, 1] > 0):
            raise ValueError(f"{scenario_id}: demand slopes must be positive.")


def _assert_unique(values: Sequence[str], field_name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate {field_name} found.")


def _build_scenarios() -> Dict[str, ScenarioSpec]:
    buybye = ScenarioSpec(
        scenario_id="buybye_autonomous",
        label="BuyBye autonomous retail",
        engine="perishable_monopoly_newsvendor",
        horizon_days=7,
        capacity_units=160,
        starting_cash=2500.0,
        skus=(
            SKU(
                sku_id="fresh_salad_bowl",
                label="Fresh salad bowl",
                cost=9.50,
                initial_price=24.00,
                shelf_life_days=1,
                salvage_value=1.00,
                max_order_units=70,
                demand=DemandSpec(intercept=126.0, slope=3.20, noise_width=16.0),
            ),
            SKU(
                sku_id="sushi_box",
                label="Sushi box",
                cost=16.00,
                initial_price=38.00,
                shelf_life_days=1,
                salvage_value=2.00,
                max_order_units=55,
                demand=DemandSpec(intercept=142.0, slope=2.45, noise_width=18.0),
            ),
            SKU(
                sku_id="cold_pressed_juice",
                label="Cold-pressed juice",
                cost=6.00,
                initial_price=18.00,
                shelf_life_days=2,
                salvage_value=0.50,
                max_order_units=80,
                demand=DemandSpec(intercept=82.0, slope=1.85, noise_width=12.0),
            ),
        ),
        metadata={
            "company_archetype": "autonomous convenience retail",
            "newsvendor_distribution": "linear demand plus bounded uniform shock",
        },
    )

    d2c = ScenarioSpec(
        scenario_id="d2c_fashion",
        label="WearClio / Disturb D2C fashion",
        engine="markdown_lifecycle",
        horizon_days=42,
        capacity_units=260,
        starting_cash=18000.0,
        skus=(
            SKU(
                sku_id="launch_dress",
                label="Launch dress",
                cost=44.00,
                initial_price=129.00,
                shelf_life_days=35,
                salvage_value=8.00,
                max_order_units=120,
                demand=DemandSpec(
                    intercept=190.0,
                    slope=0.55,
                    noise_width=20.0,
                    hype_decay=0.055,
                    slope_growth_per_day=0.035,
                ),
            ),
            SKU(
                sku_id="capsule_tee",
                label="Capsule tee",
                cost=17.00,
                initial_price=49.00,
                shelf_life_days=28,
                salvage_value=3.00,
                max_order_units=220,
                demand=DemandSpec(
                    intercept=245.0,
                    slope=2.60,
                    noise_width=30.0,
                    hype_decay=0.045,
                    slope_growth_per_day=0.050,
                ),
            ),
            SKU(
                sku_id="limited_jacket",
                label="Limited jacket",
                cost=78.00,
                initial_price=219.00,
                shelf_life_days=42,
                salvage_value=20.00,
                max_order_units=75,
                demand=DemandSpec(
                    intercept=130.0,
                    slope=0.30,
                    noise_width=14.0,
                    hype_decay=0.035,
                    slope_growth_per_day=0.025,
                ),
            ),
        ),
        metadata={
            "company_archetype": "short-cycle D2C fashion",
            "markdown_days": (0, 14, 28),
            "final_clearance_day": 28,
        },
    )

    baco = ScenarioSpec(
        scenario_id="baco_premium",
        label="Seja Baco premium beverages",
        engine="premium_bundling_price_discrimination",
        horizon_days=14,
        capacity_units=120,
        starting_cash=20000.0,
        skus=(
            SKU(
                sku_id="natural_wine",
                label="Natural wine",
                cost=58.00,
                initial_price=149.00,
                shelf_life_days=180,
                max_order_units=60,
                demand=DemandSpec(intercept=82.0, slope=0.35, noise_width=6.0),
            ),
            SKU(
                sku_id="aged_cachaca",
                label="Aged cachaca",
                cost=72.00,
                initial_price=179.00,
                shelf_life_days=365,
                max_order_units=45,
                demand=DemandSpec(intercept=70.0, slope=0.24, noise_width=5.0),
            ),
            SKU(
                sku_id="reserve_sparkling",
                label="Reserve sparkling",
                cost=96.00,
                initial_price=249.00,
                shelf_life_days=240,
                max_order_units=40,
                demand=DemandSpec(intercept=58.0, slope=0.16, noise_width=4.0),
            ),
        ),
        bundles=(
            BundleSpec(
                bundle_id="curated_duo",
                label="Curated duo",
                components=(
                    BundleComponent("natural_wine", 1),
                    BundleComponent("reserve_sparkling", 1),
                ),
                initial_price=379.00,
                tier="gift",
                demand=DemandSpec(intercept=74.0, slope=0.13, noise_width=5.0),
            ),
            BundleSpec(
                bundle_id="tasting_trio",
                label="Tasting trio",
                components=(
                    BundleComponent("natural_wine", 1),
                    BundleComponent("aged_cachaca", 1),
                    BundleComponent("reserve_sparkling", 1),
                ),
                initial_price=489.00,
                tier="collector",
                demand=DemandSpec(intercept=62.0, slope=0.085, noise_width=4.0),
            ),
        ),
        metadata={
            "company_archetype": "premium beverage curation",
            "bundling": "component-constrained curated bundles",
        },
    )

    return {
        buybye.scenario_id: buybye,
        d2c.scenario_id: d2c,
        baco.scenario_id: baco,
    }


SCENARIOS = _build_scenarios()
validate_catalog(SCENARIOS)

