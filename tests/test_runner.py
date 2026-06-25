import unittest

from runner import (
    OracleAgent,
    build_structural_diagnostic,
    run_episode,
    run_multi_seed,
    sample_scenarios_for_seeds,
)


class RunnerTest(unittest.TestCase):
    def test_sample_scenarios_covers_all_mvp_scenarios(self) -> None:
        assignment = sample_scenarios_for_seeds(
            seeds=[11, 22, 33],
            scenario_ids=["buybye_autonomous", "d2c_fashion", "baco_premium"],
            rng_seed=7,
        )

        self.assertEqual(
            set(assignment.values()),
            {"buybye_autonomous", "d2c_fashion", "baco_premium"},
        )

    def test_run_episode_returns_summary_and_step_records(self) -> None:
        episode = run_episode(
            agent=OracleAgent(),
            scenario="buybye_autonomous",
            seed=101,
            stochastic=False,
        )

        self.assertEqual(episode["scenario_id"], "buybye_autonomous")
        self.assertEqual(len(episode["steps"]), 7)
        self.assertIn("summary", episode)
        self.assertIn("diagnostic", episode["summary"])
        self.assertAlmostEqual(episode["summary"]["mean_price_efficiency"], 1.0)

    def test_run_multi_seed_aggregates_by_scenario(self) -> None:
        batch = run_multi_seed(
            agent=OracleAgent(),
            seeds=[1, 2, 3],
            scenario_ids=["buybye_autonomous", "d2c_fashion", "baco_premium"],
            stochastic=False,
            scenario_rng_seed=1,
        )

        self.assertEqual(len(batch["episodes"]), 3)
        self.assertEqual(
            set(batch["summary"]["scenarios"]),
            {"buybye_autonomous", "d2c_fashion", "baco_premium"},
        )
        self.assertAlmostEqual(batch["summary"]["mean_price_efficiency"], 1.0)

    def test_diagnostics_identify_buybye_inventory_failure(self) -> None:
        diagnostic = build_structural_diagnostic(
            {
                "scenario_id": "buybye_autonomous",
                "total_expired_units": [30.0, 0.0, 0.0],
                "total_orders": [100.0, 0.0, 0.0],
                "total_lost_sales": [0.0, 0.0, 0.0],
                "mean_abs_order_gap": 10.0,
            }
        )

        self.assertEqual(diagnostic["status"], "fail")
        self.assertEqual(diagnostic["failure_mode"], "gerir estoque")

    def test_diagnostics_identify_d2c_markdown_failure(self) -> None:
        diagnostic = build_structural_diagnostic(
            {
                "scenario_id": "d2c_fashion",
                "final_inventory": [8.0, 2.0, 0.0],
                "mean_late_markdown": 0.0,
                "mean_late_relative_price_gap": 0.25,
            }
        )

        self.assertEqual(diagnostic["status"], "fail")
        self.assertEqual(diagnostic["failure_mode"], "liquidar moda")

    def test_diagnostics_identify_baco_bundle_pricing_failure(self) -> None:
        diagnostic = build_structural_diagnostic(
            {
                "scenario_id": "baco_premium",
                "mean_bundle_relative_price_gap": 0.22,
                "total_bundle_sales": [2.0, 1.0],
            }
        )

        self.assertEqual(diagnostic["status"], "fail")
        self.assertEqual(diagnostic["failure_mode"], "precificar pacotes")


if __name__ == "__main__":
    unittest.main()

