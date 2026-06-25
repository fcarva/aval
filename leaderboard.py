"""Comparative certification: rank a roster of agents by AVAL Score.

This is the spec's v1 centerpiece - the first board that ranks agents by
structural economic competence rather than by accumulated profit. Every agent
faces the same seeds and the same scenario assignment, so demand realizations are
identical across the roster and the comparison is apples-to-apples.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from runner import (
    FixedMarkupAgent,
    NaiveAgent,
    OracleAgent,
    RandomAgent,
    run_multi_seed,
)


AgentFactory = Callable[[], Any]

# Display labels for the built-in roster. Anything not listed falls back to its
# registry key.
AGENT_LABELS = {
    "oracle": "OracleAgent (otimo teorico)",
    "naive": "NaiveAgent (demo ingenuo)",
    "fixed_markup": "FixedMarkup (cost-plus)",
    "random": "Random (volatil)",
    "heuristic_llm": "LLM heuristico (contrato JSON)",
}

DEFAULT_SEEDS = (101, 202, 303, 404, 505, 606)


def default_roster() -> dict[str, AgentFactory]:
    """The built-in, offline agent roster (numpy-only core)."""

    return {
        "oracle": OracleAgent,
        "naive": NaiveAgent,
        "fixed_markup": FixedMarkupAgent,
        "random": lambda: RandomAgent(seed=0),
    }


def _entry_from_summary(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "label": AGENT_LABELS.get(name, name),
        "aval_score": float(summary.get("mean_aval_score", 0.0)),
        "grade": str(summary.get("grade", "F")),
        "verdict": str(summary.get("verdict", "REPROVADO")),
        "survival_rate": float(summary.get("survival_rate", 0.0)),
        "mean_price_efficiency": float(summary.get("mean_price_efficiency", 0.0)),
        "mean_structural_r_squared": float(
            summary.get("mean_structural_r_squared", 0.0)
        ),
        "total_profit": float(summary.get("total_profit", 0.0)),
        "episodes": int(summary.get("episodes", 0)),
    }


def _rank_key(entry: Mapping[str, Any]) -> tuple[float, float, float]:
    # Score first; break ties on survival, then pricing efficiency. Profit is
    # deliberately NOT a tiebreaker - the whole point is that profit is not
    # competence.
    return (
        entry["aval_score"],
        entry["survival_rate"],
        entry["mean_price_efficiency"],
    )


def run_leaderboard(
    roster: Mapping[str, AgentFactory] | None = None,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    scenario_ids: Sequence[str] | None = None,
    stochastic: bool = True,
    scenario_rng_seed: int = 0,
) -> dict[str, Any]:
    """Run every agent over the same seeds/scenarios and rank them by AVAL Score."""

    roster = dict(roster) if roster is not None else default_roster()
    if not roster:
        raise ValueError("roster cannot be empty.")
    if not seeds:
        raise ValueError("seeds cannot be empty.")

    seeds = [int(seed) for seed in seeds]
    scenario_ids = list(scenario_ids) if scenario_ids is not None else None

    entries: list[dict[str, Any]] = []
    for name, factory in roster.items():
        batch = run_multi_seed(
            agent=factory(),
            seeds=seeds,
            scenario_ids=scenario_ids,
            stochastic=stochastic,
            scenario_rng_seed=scenario_rng_seed,
        )
        entries.append(_entry_from_summary(name, batch["summary"]))

    entries.sort(key=_rank_key, reverse=True)
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    return {
        "seeds": seeds,
        "scenario_ids": scenario_ids,
        "stochastic": bool(stochastic),
        "scenario_rng_seed": int(scenario_rng_seed),
        "entries": entries,
    }


def format_board(board: Mapping[str, Any]) -> str:
    """Plain-text board for CLI output."""

    lines = [
        f"{'#':>2}  {'Agent':28} {'Score':>6} {'Grade':>5} {'Verdict':>10} "
        f"{'Surv':>5} {'Eff':>5} {'StrR2':>6} {'Profit':>12}",
    ]
    for entry in board["entries"]:
        lines.append(
            f"{entry['rank']:>2}  {entry['label'][:28]:28} "
            f"{entry['aval_score']:6.1f} {entry['grade']:>5} {entry['verdict']:>10} "
            f"{entry['survival_rate'] * 100:4.0f}% {entry['mean_price_efficiency']:5.2f} "
            f"{entry['mean_structural_r_squared']:6.2f} {entry['total_profit']:12.0f}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    from report import write_leaderboard_report

    board = run_leaderboard(stochastic=True)
    print(format_board(board))
    path = write_leaderboard_report(board, "aval_leaderboard.html")
    print(f"\nWrote {path}")
