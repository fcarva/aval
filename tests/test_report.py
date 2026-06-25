import tempfile
import unittest
from pathlib import Path

from report import render_html_report, write_html_report
from runner import build_structural_diagnostic, summarize_batch


def episode_summary(scenario_id, seed, diagnostic_inputs):
    summary = {
        "scenario_id": scenario_id,
        "seed": seed,
        "days_run": 1,
        "total_profit": 10.0,
        "mean_price_efficiency": 0.75,
        "mean_abs_relative_price_gap": 0.2,
        "mean_abs_order_gap": 3.0,
        **diagnostic_inputs,
    }
    summary["diagnostic"] = build_structural_diagnostic(summary)
    return {"scenario_id": scenario_id, "seed": seed, "summary": summary, "steps": []}


class ReportTest(unittest.TestCase):
    def test_report_renders_structural_diagnostic_failure_modes(self) -> None:
        episodes = [
            episode_summary(
                "buybye_autonomous",
                1,
                {
                    "total_expired_units": [25.0, 0.0, 0.0],
                    "total_orders": [100.0, 0.0, 0.0],
                    "total_lost_sales": [0.0, 0.0, 0.0],
                },
            ),
            episode_summary(
                "d2c_fashion",
                2,
                {
                    "final_inventory": [5.0, 4.0, 3.0],
                    "mean_late_markdown": 0.0,
                    "mean_late_relative_price_gap": 0.3,
                },
            ),
            episode_summary(
                "baco_premium",
                3,
                {
                    "mean_bundle_relative_price_gap": 0.2,
                    "total_bundle_sales": [1.0, 1.0],
                },
            ),
        ]
        batch = {"episodes": episodes, "summary": summarize_batch(episodes)}
        html = render_html_report(batch)

        self.assertIn("Diagn&oacute;stico estrutural", html)
        self.assertIn("gerir estoque", html)
        self.assertIn("liquidar moda", html)
        self.assertIn("precificar pacotes", html)
        self.assertIn("AVAL Parecer", html)

    def test_write_html_report_creates_file(self) -> None:
        episodes = [
            episode_summary(
                "baco_premium",
                3,
                {
                    "mean_bundle_relative_price_gap": 0.0,
                    "total_bundle_sales": [1.0, 1.0],
                },
            )
        ]
        batch = {"episodes": episodes, "summary": summarize_batch(episodes)}
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "aval_parecer.html"
            returned_path = write_html_report(batch, output_path)

            self.assertEqual(Path(returned_path), output_path)
            self.assertTrue(output_path.exists())
            self.assertIn("AVAL Parecer", output_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

