import unittest

from leaderboard import default_roster, format_board, run_leaderboard
from runner import NaiveAgent, OracleAgent


class LeaderboardTest(unittest.TestCase):
    def test_default_roster_has_the_baselines(self) -> None:
        self.assertEqual(
            set(default_roster()),
            {"oracle", "naive", "fixed_markup", "random"},
        )

    def test_run_leaderboard_ranks_oracle_first(self) -> None:
        board = run_leaderboard(
            seeds=[1, 2, 3, 4, 5, 6],
            scenario_ids=["buybye_autonomous", "d2c_fashion", "baco_premium"],
            stochastic=False,
            scenario_rng_seed=0,
        )
        entries = board["entries"]
        self.assertEqual(len(entries), 4)
        self.assertEqual(entries[0]["name"], "oracle")
        self.assertEqual(entries[0]["verdict"], "APROVADO")
        # Ranks are dense and ordered, scores are non-increasing.
        self.assertEqual([e["rank"] for e in entries], [1, 2, 3, 4])
        scores = [e["aval_score"] for e in entries]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_run_leaderboard_is_apples_to_apples(self) -> None:
        # Every agent runs the same seeds and scenario assignment, so a fresh run
        # with the same inputs is reproducible.
        kwargs = dict(
            seeds=[7, 8, 9],
            scenario_ids=["buybye_autonomous"],
            stochastic=False,
            scenario_rng_seed=3,
        )
        first = run_leaderboard(**kwargs)
        second = run_leaderboard(**kwargs)
        self.assertEqual(
            [e["name"] for e in first["entries"]],
            [e["name"] for e in second["entries"]],
        )
        self.assertAlmostEqual(
            first["entries"][0]["aval_score"], second["entries"][0]["aval_score"]
        )

    def test_custom_roster_and_format(self) -> None:
        board = run_leaderboard(
            roster={"oracle": OracleAgent, "naive": NaiveAgent},
            seeds=[11, 22],
            scenario_ids=["baco_premium"],
            stochastic=False,
        )
        names = {entry["name"] for entry in board["entries"]}
        self.assertEqual(names, {"oracle", "naive"})
        text = format_board(board)
        self.assertIn("OracleAgent", text)

    def test_empty_inputs_raise(self) -> None:
        with self.assertRaises(ValueError):
            run_leaderboard(seeds=[])
        with self.assertRaises(ValueError):
            run_leaderboard(roster={}, seeds=[1])


if __name__ == "__main__":
    unittest.main()
