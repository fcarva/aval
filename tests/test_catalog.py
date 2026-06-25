import unittest

import numpy as np

from catalog import (
    DEMAND_PARAM_COLUMNS,
    SCENARIOS,
    ScenarioSpec,
    demand_at_price,
    expected_demand_vector,
    validate_catalog,
)


class CatalogTest(unittest.TestCase):
    def test_catalog_has_three_mvp_scenarios(self) -> None:
        self.assertEqual(
            set(SCENARIOS),
            {"buybye_autonomous", "d2c_fashion", "baco_premium"},
        )
        for scenario in SCENARIOS.values():
            self.assertIsInstance(scenario, ScenarioSpec)
            self.assertGreaterEqual(len(scenario.skus), 3)

    def test_catalog_validates(self) -> None:
        validate_catalog(SCENARIOS)

    def test_matrices_have_consistent_shapes(self) -> None:
        for scenario in SCENARIOS.values():
            matrices = scenario.to_matrices()
            offer_count = len(scenario.offer_ids(include_bundles=True))

            self.assertEqual(matrices["demand_param_columns"], DEMAND_PARAM_COLUMNS)
            self.assertEqual(matrices["costs"].shape, (offer_count,))
            self.assertEqual(matrices["initial_prices"].shape, (offer_count,))
            self.assertEqual(matrices["shelf_life_days"].shape, (offer_count,))
            self.assertEqual(matrices["demand_params"].shape, (offer_count, 5))
            self.assertEqual(matrices["is_bundle"].shape, (offer_count,))
            self.assertTrue(np.all(matrices["costs"] > 0.0))
            self.assertTrue(np.all(matrices["initial_prices"] > 0.0))
            self.assertTrue(np.all(matrices["shelf_life_days"] > 0))

    def test_buybye_contains_newsvendor_primitives(self) -> None:
        scenario = SCENARIOS["buybye_autonomous"]
        matrices = scenario.to_matrices()

        self.assertEqual(scenario.engine, "perishable_monopoly_newsvendor")
        self.assertIsNotNone(scenario.capacity_units)
        self.assertTrue(np.all(matrices["salvage_values"] < matrices["costs"]))
        self.assertTrue(np.all(matrices["max_order_units"][: len(scenario.skus)] > 0))
        self.assertTrue(
            np.all(
                expected_demand_vector(
                    scenario,
                    matrices["initial_prices"],
                    include_bundles=True,
                )
                > 0
            )
        )

    def test_d2c_hype_decays_and_price_sensitivity_rises(self) -> None:
        scenario = SCENARIOS["d2c_fashion"]
        self.assertEqual(scenario.engine, "markdown_lifecycle")

        for sku in scenario.skus:
            intercept_day_0, slope_day_0 = sku.demand.params_at_day(day=0)
            intercept_day_28, slope_day_28 = sku.demand.params_at_day(day=28)
            self.assertLess(intercept_day_28, intercept_day_0)
            self.assertGreater(slope_day_28, slope_day_0)
            self.assertLess(
                demand_at_price(sku.demand, sku.initial_price, day=28),
                demand_at_price(sku.demand, sku.initial_price, day=0),
            )

    def test_premium_bundles_reference_existing_skus(self) -> None:
        scenario = SCENARIOS["baco_premium"]
        matrices = scenario.to_matrices()
        component_matrix = matrices["bundle_component_matrix"]

        self.assertEqual(scenario.engine, "premium_bundling_price_discrimination")
        self.assertEqual(component_matrix.shape, (len(scenario.bundles), len(scenario.skus)))
        self.assertTrue(np.all(component_matrix.sum(axis=1) >= 2))
        self.assertTrue(np.all(matrices["is_bundle"][-len(scenario.bundles) :]))

        sku_cost_by_id = {sku.sku_id: sku.cost for sku in scenario.skus}
        for bundle in scenario.bundles:
            expected_cost = sum(
                sku_cost_by_id[component.sku_id] * component.quantity
                for component in bundle.components
            )
            self.assertAlmostEqual(scenario.bundle_cost(bundle), expected_cost)
            self.assertGreater(bundle.initial_price, expected_cost)


if __name__ == "__main__":
    unittest.main()

