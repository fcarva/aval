import unittest

from scoring import (
    aggregate_certificate,
    aval_certificate,
    grade_for_score,
    layer_scores,
    verdict_for_score,
)


class ScoringTest(unittest.TestCase):
    def test_perfect_summary_scores_high_and_approves(self) -> None:
        summary = {
            "mean_price_efficiency": 1.0,
            "mean_order_accuracy": 1.0,
            "mean_structural_r_squared": 1.0,
            "flag_rate": 0.0,
            "survived": True,
            "price_incoherence": 0.0,
        }
        cert = aval_certificate(summary)
        self.assertAlmostEqual(cert["aval_score"], 100.0)
        self.assertEqual(cert["grade"], "A+")
        self.assertEqual(cert["verdict"], "APROVADO")

    def test_zero_summary_scores_floor_and_reproves(self) -> None:
        summary = {
            "mean_price_efficiency": 0.0,
            "mean_order_accuracy": 0.0,
            "mean_structural_r_squared": 0.0,
            "flag_rate": 1.0,
            "survived": False,
            "price_incoherence": 1.0,
        }
        cert = aval_certificate(summary)
        self.assertEqual(cert["aval_score"], 0.0)
        self.assertEqual(cert["grade"], "F")
        self.assertEqual(cert["verdict"], "REPROVADO")

    def test_layer_scores_are_clipped_to_unit_interval(self) -> None:
        layers = layer_scores(
            {
                "mean_price_efficiency": 1.4,
                "mean_order_accuracy": -0.2,
                "mean_structural_r_squared": float("nan"),
                "flag_rate": 0.3,
                "survived": True,
                "price_incoherence": 0.5,
            }
        )
        self.assertEqual(layers["pricing"], 1.0)
        self.assertEqual(layers["inventory"], 0.0)
        self.assertEqual(layers["structural"], 0.0)
        self.assertAlmostEqual(layers["flags"], 0.7)
        self.assertAlmostEqual(layers["coherence"], 0.6 + 0.4 * 0.5)

    def test_grade_and_verdict_bands(self) -> None:
        self.assertEqual(grade_for_score(95.0), "A+")
        self.assertEqual(grade_for_score(82.0), "B+")
        self.assertEqual(grade_for_score(49.9), "F")
        self.assertEqual(verdict_for_score(80.0), "APROVADO")
        self.assertEqual(verdict_for_score(60.0), "RESSALVA")
        self.assertEqual(verdict_for_score(59.9), "REPROVADO")

    def test_aggregate_certificate_averages_scores(self) -> None:
        cert = aggregate_certificate([90.0, 70.0])
        self.assertAlmostEqual(cert["aval_score"], 80.0)
        self.assertEqual(cert["verdict"], "APROVADO")
        empty = aggregate_certificate([])
        self.assertEqual(empty["grade"], "F")


if __name__ == "__main__":
    unittest.main()
