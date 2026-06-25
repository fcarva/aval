import unittest

import numpy as np

from catalog import SCENARIOS
from metrics import (
    clipped_uniform_quantile,
    cost_pass_through,
    demand_parameters_at_day,
    implied_lerner_elasticity,
    monopoly_price,
    newsvendor_critical_fractile,
    newsvendor_quantity,
    newsvendor_quantities,
    oracle_monopoly_outcome,
    oracle_price_vector,
    price_efficiency,
    structural_recovery,
)


class MetricsTest(unittest.TestCase):
    def test_monopoly_price_uses_closed_form(self) -> None:
        self.assertAlmostEqual(monopoly_price(intercept=100.0, slope=2.0, cost=10.0), 30.0)

    def test_oracle_price_vector_matches_formula_for_catalog(self) -> None:
        scenario = SCENARIOS["buybye_autonomous"]
        matrices = scenario.to_matrices()
        intercepts, slopes = demand_parameters_at_day(scenario)
        prices = oracle_price_vector(scenario)
        expected = (intercepts + slopes * matrices["costs"]) / (2.0 * slopes)

        np.testing.assert_allclose(prices, expected)

    def test_d2c_oracle_prices_fall_as_hype_decays(self) -> None:
        scenario = SCENARIOS["d2c_fashion"]
        day_0_prices = oracle_price_vector(scenario, day=0, include_bundles=False)
        day_28_prices = oracle_price_vector(scenario, day=28, include_bundles=False)

        self.assertTrue(np.all(day_28_prices < day_0_prices))

    def test_newsvendor_critical_fractile_and_quantile_are_closed_form(self) -> None:
        fractile = newsvendor_critical_fractile(price=10.0, cost=4.0, salvage_value=1.0)
        self.assertAlmostEqual(fractile, 6.0 / 9.0)
        self.assertAlmostEqual(
            clipped_uniform_quantile(center=50.0, half_width=15.0, probability=fractile),
            55.0,
        )
        self.assertAlmostEqual(
            newsvendor_quantity(
                intercept=100.0,
                slope=5.0,
                price=10.0,
                cost=4.0,
                salvage_value=1.0,
                noise_width=15.0,
            ),
            55.0,
        )

    def test_buybye_newsvendor_oracle_quantities_are_feasible_quantiles(self) -> None:
        scenario = SCENARIOS["buybye_autonomous"]
        prices = oracle_price_vector(scenario, include_bundles=False)
        quantities = newsvendor_quantities(scenario, prices=prices)
        intercepts, slopes = demand_parameters_at_day(scenario, include_bundles=False)
        noise_width = scenario.to_matrices(include_bundles=False)["demand_params"][:, 2]
        upper_support = intercepts - slopes * prices + noise_width

        self.assertTrue(np.all(quantities >= 0.0))
        self.assertTrue(np.all(quantities <= upper_support))

    def test_structural_recovery_reports_pass_through_and_elasticity(self) -> None:
        costs = np.array([2.0, 4.0, 6.0])
        prices = 10.0 + 2.0 * costs
        self.assertAlmostEqual(cost_pass_through(costs, prices), 2.0)
        np.testing.assert_allclose(
            implied_lerner_elasticity(prices, costs),
            -prices / (prices - costs),
        )

        scenario = SCENARIOS["baco_premium"]
        oracle_prices = oracle_price_vector(scenario)
        recovery = structural_recovery(scenario, oracle_prices)
        np.testing.assert_allclose(recovery["absolute_price_gap"], 0.0)
        np.testing.assert_allclose(recovery["oracle_cost_pass_through"], 0.5)

    def test_price_efficiency_is_one_at_oracle_prices(self) -> None:
        scenario = SCENARIOS["baco_premium"]
        oracle_prices = oracle_price_vector(scenario)
        efficiency = price_efficiency(scenario, oracle_prices)

        self.assertAlmostEqual(efficiency["efficiency_ratio"], 1.0)

    def test_oracle_monopoly_outcome_has_positive_profit(self) -> None:
        outcome = oracle_monopoly_outcome("baco_premium")

        self.assertGreater(outcome["total_profit"], 0.0)
        self.assertEqual(len(outcome["offer_ids"]), 5)


if __name__ == "__main__":
    unittest.main()

