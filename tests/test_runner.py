import unittest

from runner import (
    FixedMarkupAgent,
    NaiveAgent,
    OracleAgent,
    RandomAgent,
    build_structural_diagnostic,
    run_episode,
    run_multi_seed,
    sample_scenarios_for_seeds,
)


class _RuinousAgent:
    """Prices everything at the floor and orders to the limit: guaranteed loss."""

    def act(self, obs):
        offer_ids = [str(item) for item in obs["offer_ids"]]
        sku_ids = [str(item) for item in obs["sku_ids"]]
        return {
            "prices": {offer_id: 0.01 for offer_id in offer_ids},
            "orders": {sku_id: 1_000.0 for sku_id in sku_ids},
            "markdowns": {offer_id: 0.0 for offer_id in offer_ids},
        }


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

    def test_naive_agent_demo_exposes_all_structural_failures(self) -> None:
        batch = run_multi_seed(
            agent=NaiveAgent(),
            seeds=[101, 202, 303],
            scenario_ids=["buybye_autonomous", "d2c_fashion", "baco_premium"],
            stochastic=False,
            scenario_rng_seed=0,
        )

        failures = {
            episode["scenario_id"]: episode["summary"]["diagnostic"]["failure_mode"]
            for episode in batch["episodes"]
        }
        self.assertLess(batch["summary"]["mean_price_efficiency"], 0.90)
        self.assertEqual(
            failures,
            {
                "buybye_autonomous": "gerir estoque",
                "d2c_fashion": "liquidar moda",
                "baco_premium": "precificar pacotes",
            },
        )

    def test_episode_summary_carries_certificate(self) -> None:
        episode = run_episode(OracleAgent(), "baco_premium", seed=5, stochastic=False)
        summary = episode["summary"]
        for key in ("aval_score", "grade", "verdict", "certificate", "survived"):
            self.assertIn(key, summary)
        self.assertGreaterEqual(summary["aval_score"], 0.0)
        self.assertLessEqual(summary["aval_score"], 100.0)

    def test_baseline_agents_produce_valid_runnable_actions(self) -> None:
        for agent in (FixedMarkupAgent(), RandomAgent(seed=1)):
            episode = run_episode(agent, "buybye_autonomous", seed=3, stochastic=False)
            self.assertEqual(len(episode["steps"]), 7)
            self.assertIn("aval_score", episode["summary"])

    def test_aval_score_ranks_oracle_above_naive(self) -> None:
        seeds = [1, 2, 3, 4, 5, 6]
        ids = ["buybye_autonomous", "d2c_fashion", "baco_premium"]
        oracle = run_multi_seed(OracleAgent(), seeds, scenario_ids=ids, stochastic=True)
        naive = run_multi_seed(NaiveAgent(), seeds, scenario_ids=ids, stochastic=True)
        self.assertGreater(
            oracle["summary"]["mean_aval_score"], naive["summary"]["mean_aval_score"]
        )
        self.assertEqual(oracle["summary"]["verdict"], "APROVADO")

    def test_ruinous_agent_fails_survival(self) -> None:
        episode = run_episode(_RuinousAgent(), "buybye_autonomous", seed=1, stochastic=False)
        summary = episode["summary"]
        self.assertFalse(summary["survived"])
        # survival 0 -> coherence drops to the incoherence component only.
        self.assertAlmostEqual(summary["certificate"]["layers"]["coherence"], 0.4)
        self.assertLess(summary["aval_score"], 60.0)

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
