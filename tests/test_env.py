import unittest

import numpy as np

from catalog import SCENARIOS
from env import RetailEnv, StepResult


class RetailEnvTest(unittest.TestCase):
    def test_step_result_can_be_unpacked_like_gym_tuple(self) -> None:
        env = RetailEnv("buybye_autonomous", stochastic=False)
        result = env.step({"prices": [999.0, 999.0, 999.0]})

        self.assertIsInstance(result, StepResult)
        observation, reward, done, info = result
        self.assertEqual(observation["day"], 1)
        self.assertIsInstance(reward, float)
        self.assertFalse(done)
        self.assertIn("profit", info)

    def test_d2c_transition_uses_day_specific_hype_decay(self) -> None:
        scenario = SCENARIOS["d2c_fashion"]
        prices = scenario.to_matrices()["initial_prices"]

        env_day_0 = RetailEnv(scenario, stochastic=False)
        env_day_0.reset(initial_day=0)
        day_0 = env_day_0.step({"prices": prices}).info["expected_demand"]

        env_day_28 = RetailEnv(scenario, stochastic=False)
        env_day_28.reset(initial_day=28)
        day_28 = env_day_28.step({"prices": prices}).info["expected_demand"]

        self.assertTrue(np.all(day_28 < day_0))

    def test_premium_bundle_sales_consume_component_inventory(self) -> None:
        env = RetailEnv("baco_premium", stochastic=False)
        env.reset(initial_inventory={"natural_wine": 1.0, "reserve_sparkling": 1.0})
        result = env.step()

        np.testing.assert_allclose(result.info["bundle_sales"], [1.0, 0.0])
        np.testing.assert_allclose(result.info["inventory_end"], [0.0, 0.0, 0.0])

    def test_perishable_inventory_expires_with_salvage_recovery(self) -> None:
        env = RetailEnv("buybye_autonomous", stochastic=False)
        env.reset(initial_inventory={"fresh_salad_bowl": 5.0})
        result = env.step({"prices": [999.0, 999.0, 999.0]})

        self.assertAlmostEqual(result.info["expired_units"][0], 5.0)
        self.assertAlmostEqual(result.info["salvage_revenue"], 5.0)
        self.assertAlmostEqual(result.info["inventory_end"][0], 0.0)

    def test_orders_are_clipped_by_capacity_and_sku_limits(self) -> None:
        env = RetailEnv("buybye_autonomous", stochastic=False)
        result = env.step(
            {
                "orders": {
                    "fresh_salad_bowl": 1_000.0,
                    "sushi_box": 1_000.0,
                    "cold_pressed_juice": 1_000.0,
                },
                "prices": [999.0, 999.0, 999.0],
            }
        )

        np.testing.assert_allclose(result.info["orders"], [70.0, 55.0, 35.0])
        self.assertLessEqual(result.info["orders"].sum(), 160.0)

    def test_cash_is_tracked_when_scenario_sets_starting_cash(self) -> None:
        env = RetailEnv("buybye_autonomous", stochastic=False)
        observation = env.observation()
        self.assertEqual(observation["starting_cash"], 2500.0)
        self.assertEqual(observation["cash"], 2500.0)
        self.assertFalse(observation["bankrupt"])

    def test_ruinous_overordering_triggers_bankruptcy_flag(self) -> None:
        env = RetailEnv("buybye_autonomous", stochastic=False)
        # Order to the physical limit at an unsellable price: pure cash burn.
        ruinous = {
            "orders": {
                "fresh_salad_bowl": 1_000.0,
                "sushi_box": 1_000.0,
                "cold_pressed_juice": 1_000.0,
            },
            "prices": [999.0, 999.0, 999.0],
        }
        bankrupt = False
        for _ in range(3):
            if env.day >= env.scenario.horizon_days:
                break
            result = env.step(ruinous)
            bankrupt = bankrupt or result.info["bankrupt"]
        self.assertTrue(bankrupt)
        self.assertLess(env.cash, 0.0)

    def test_markdowns_update_prices_before_demand(self) -> None:
        env = RetailEnv("d2c_fashion", stochastic=False)
        initial_prices = env.observation()["current_prices"]
        result = env.step({"markdowns": [0.25, 0.25, 0.25]})

        np.testing.assert_allclose(result.info["prices"], initial_prices * 0.75)


if __name__ == "__main__":
    unittest.main()

