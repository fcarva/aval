import json
import unittest

from agents import AgentParseError, LLMAgent, LLMRequest, ModelClientNotConfigured
from env import RetailEnv


def response_for_obs(obs, price_offset=0.0):
    return json.dumps(
        {
            "precos": {
                str(offer_id): float(price) + price_offset
                for offer_id, price in zip(obs["offer_ids"], obs["current_prices"])
            },
            "pedidos": {str(sku_id): 1.0 for sku_id in obs["sku_ids"]},
            "markdowns": {str(offer_id): 0.0 for offer_id in obs["offer_ids"]},
        }
    )


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.last_request = None

    def complete(self, request):
        self.last_request = request
        return self.response


class LLMAgentTest(unittest.TestCase):
    def test_render_includes_state_and_required_json_contract(self) -> None:
        obs = RetailEnv("baco_premium", stochastic=False).observation()
        rendered = LLMAgent().render(obs)

        self.assertIn("Scenario: baco_premium", rendered)
        self.assertIn("natural_wine", rendered)
        self.assertIn("curated_duo", rendered)
        self.assertIn("1.00 x natural_wine", rendered)
        self.assertIn('"precos"', rendered)
        self.assertIn('"pedidos"', rendered)
        self.assertIn('"markdowns"', rendered)

    def test_parse_valid_json_returns_env_action(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()
        decision = LLMAgent().parse(response_for_obs(obs), obs)
        action = decision.to_env_action()

        self.assertEqual(set(action), {"prices", "orders", "markdowns"})
        self.assertEqual(set(action["prices"]), set(obs["offer_ids"]))
        self.assertEqual(set(action["orders"]), set(obs["sku_ids"]))
        self.assertTrue(all(value > 0.0 for value in action["prices"].values()))
        self.assertTrue(all(value == 1.0 for value in action["orders"].values()))

    def test_parse_extracts_json_from_markdown_fence(self) -> None:
        obs = RetailEnv("d2c_fashion", stochastic=False).observation()
        text = "```json\n" + response_for_obs(obs) + "\n```"
        decision = LLMAgent().parse(text, obs)

        self.assertEqual(set(decision.precos), set(obs["offer_ids"]))

    def test_parse_accepts_vector_values_in_observation_order(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()
        payload = {
            "precos": [25.0, 35.0, 18.0],
            "pedidos": [2.0, 3.0, 4.0],
            "markdowns": [0.0, 0.1, 0.2],
        }
        decision = LLMAgent().parse(json.dumps(payload), obs)

        self.assertEqual(decision.precos["fresh_salad_bowl"], 25.0)
        self.assertEqual(decision.pedidos["sushi_box"], 3.0)
        self.assertEqual(decision.markdowns["cold_pressed_juice"], 0.2)

    def test_parse_rejects_missing_required_keys(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()
        payload = {"precos": {}, "pedidos": {}}

        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

    def test_parse_rejects_unknown_or_missing_ids(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()
        payload = json.loads(response_for_obs(obs))
        payload["precos"]["ghost_sku"] = 10.0

        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

        payload = json.loads(response_for_obs(obs))
        del payload["precos"]["sushi_box"]
        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

    def test_parse_rejects_invalid_bounds(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()
        payload = json.loads(response_for_obs(obs))
        payload["precos"]["sushi_box"] = 0.0
        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

        payload = json.loads(response_for_obs(obs))
        payload["markdowns"]["sushi_box"] = 1.0
        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

        payload = json.loads(response_for_obs(obs))
        payload["pedidos"]["sushi_box"] = -1.0
        with self.assertRaises(AgentParseError):
            LLMAgent().parse(json.dumps(payload), obs)

    def test_act_calls_provider_neutral_client(self) -> None:
        obs = RetailEnv("baco_premium", stochastic=False).observation()
        client = FakeClient(response_for_obs(obs, price_offset=1.0))
        agent = LLMAgent(client=client, model="test-model", temperature=0.2)
        action = agent.act(obs)

        self.assertIsInstance(client.last_request, LLMRequest)
        self.assertEqual(client.last_request.model, "test-model")
        self.assertEqual(client.last_request.temperature, 0.2)
        self.assertIn("baco_premium", client.last_request.user_prompt)
        self.assertEqual(set(action), {"prices", "orders", "markdowns"})

    def test_act_without_client_requires_configuration(self) -> None:
        obs = RetailEnv("buybye_autonomous", stochastic=False).observation()

        with self.assertRaises(ModelClientNotConfigured):
            LLMAgent().act(obs)


if __name__ == "__main__":
    unittest.main()

