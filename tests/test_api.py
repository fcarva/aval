import importlib.util
import unittest
import warnings


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI web dependencies are not installed.")
class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings(
            "ignore",
            message=r"Using `httpx` with `starlette\.testclient` is deprecated.*",
            category=Warning,
        )
        from fastapi.testclient import TestClient
        from api import app

        cls.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_list_and_detail_scenarios(self) -> None:
        response = self.client.get("/api/scenarios")
        self.assertEqual(response.status_code, 200)
        scenarios = response.json()["scenarios"]

        self.assertEqual(len(scenarios), 3)
        detail = self.client.get("/api/scenarios/baco_premium")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertIn("curated_duo", payload["offer_ids"])
        self.assertEqual(payload["bundle_count"], 2)

    def test_run_oracle_evaluation(self) -> None:
        response = self.client.post(
            "/api/run",
            json={
                "agent": "oracle",
                "scenario_ids": ["buybye_autonomous"],
                "seeds": [101],
                "stochastic": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["summary"]["episodes"], 1)
        self.assertEqual(payload["episodes"][0]["scenario_id"], "buybye_autonomous")
        self.assertAlmostEqual(
            payload["summary"]["mean_price_efficiency"],
            1.0,
        )

    def test_run_naive_demo_exposes_failure(self) -> None:
        response = self.client.post(
            "/api/run",
            json={
                "agent": "naive",
                "scenario_ids": ["baco_premium"],
                "seeds": [101],
                "stochastic": False,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertLess(payload["summary"]["mean_price_efficiency"], 0.95)
        self.assertEqual(
            payload["episodes"][0]["summary"]["diagnostic"]["status"],
            "fail",
        )

    def test_manual_run_validates_and_executes(self) -> None:
        detail = self.client.get("/api/scenarios/buybye_autonomous").json()
        prices = {
            offer_id: price
            for offer_id, price in zip(detail["offer_ids"], detail["initial_prices"])
        }
        orders = {sku_id: 1.0 for sku_id in detail["sku_ids"]}
        markdowns = {offer_id: 0.0 for offer_id in detail["offer_ids"]}

        response = self.client.post(
            "/api/manual-run",
            json={
                "scenario_id": "buybye_autonomous",
                "seed": 101,
                "stochastic": False,
                "prices": prices,
                "orders": orders,
                "markdowns": markdowns,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["episodes"][0]["seed"], 101)

    def test_manual_run_rejects_missing_price(self) -> None:
        response = self.client.post(
            "/api/manual-run",
            json={
                "scenario_id": "buybye_autonomous",
                "prices": {},
                "orders": {},
                "markdowns": {},
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_report_endpoint_returns_url(self) -> None:
        response = self.client.post("/api/report", json={})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["url"], "/aval_parecer.html")

    def test_leaderboard_endpoint_ranks_agents(self) -> None:
        response = self.client.post(
            "/api/leaderboard",
            json={
                "seeds": [1, 2, 3],
                "scenario_ids": ["buybye_autonomous"],
                "stochastic": False,
                "include_heuristic_llm": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        entries = response.json()["entries"]
        # Oracle leads; the heuristic LLM agent is included in the roster.
        self.assertEqual(entries[0]["name"], "oracle")
        self.assertIn("heuristic_llm", {entry["name"] for entry in entries})

    def test_leaderboard_rejects_too_many_seeds(self) -> None:
        response = self.client.post(
            "/api/leaderboard", json={"seeds": list(range(201))}
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
