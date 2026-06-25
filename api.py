"""FastAPI web layer for the AVAL MVP."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from agents import LLMAgent
from catalog import SCENARIOS, ScenarioSpec, get_scenario
from env import RetailEnv
from report import DEFAULT_REPORT_PATH, write_html_report
from runner import (
    FixedMarkupAgent,
    NaiveAgent,
    OracleAgent,
    RandomAgent,
    run_multi_seed,
)

MAX_SEEDS = 200
MAX_SCENARIOS = 50

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without web deps.
    raise RuntimeError(
        "FastAPI dependencies are not installed. Run "
        "`pip install -r requirements-web.txt` before starting the web app."
    ) from exc


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
REPORT_PATH = ROOT_DIR / DEFAULT_REPORT_PATH


class RunRequest(BaseModel):
    agent: Literal[
        "naive", "oracle", "fixed_markup", "random", "heuristic_llm", "mock_llm"
    ] = "naive"
    scenario_ids: list[str] | None = Field(default=None)
    seeds: list[int] = Field(default_factory=lambda: [101, 202, 303])
    stochastic: bool = True
    scenario_rng_seed: int = 0
    generate_report: bool = False


class ManualRunRequest(BaseModel):
    scenario_id: str
    seed: int = 101
    stochastic: bool = False
    prices: dict[str, float]
    orders: dict[str, float]
    markdowns: dict[str, float] = Field(default_factory=dict)
    generate_report: bool = False


class ReportRequest(BaseModel):
    batch: dict[str, Any] | None = None


class FixedActionAgent:
    def __init__(self, action: Mapping[str, Mapping[str, float]]) -> None:
        self.action = {
            "prices": dict(action.get("prices", {})),
            "orders": dict(action.get("orders", {})),
            "markdowns": dict(action.get("markdowns", {})),
        }

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        offer_ids = tuple(str(item) for item in obs["offer_ids"])
        sku_ids = tuple(str(item) for item in obs["sku_ids"])
        return {
            "prices": {
                offer_id: float(
                    self.action["prices"].get(
                        offer_id,
                        np.asarray(obs["current_prices"], dtype=float)[idx],
                    )
                )
                for idx, offer_id in enumerate(offer_ids)
            },
            "orders": {
                sku_id: float(self.action["orders"].get(sku_id, 0.0))
                for sku_id in sku_ids
            },
            "markdowns": {
                offer_id: float(self.action["markdowns"].get(offer_id, 0.0))
                for offer_id in offer_ids
            },
        }


class HeuristicLLMAgent:
    """Honest stand-in for a real LLM, wired through the production JSON contract.

    It builds a cost-plus decision and routes it through ``LLMAgent.parse`` so the
    full prompt-render -> JSON -> validation path is exercised. It is deliberately
    naive (it ignores latent demand, like a weak model would), NOT the oracle, so
    selecting it shows a believable mediocre agent rather than a perfect one.
    """

    def __init__(self, markup: float = 0.6, restock_target: float = 12.0) -> None:
        self.markup = float(markup)
        self.restock_target = float(restock_target)
        self.parser = LLMAgent(client=lambda request: "{}")

    def act(self, obs: Mapping[str, Any]) -> Mapping[str, Any]:
        text = self._heuristic_json(obs)
        return self.parser.parse(text, obs).to_env_action()

    def render_prompt(self, obs: Mapping[str, Any]) -> str:
        return self.parser.render(obs)

    def _heuristic_json(self, obs: Mapping[str, Any]) -> str:
        offer_ids = [str(item) for item in obs["offer_ids"]]
        sku_ids = [str(item) for item in obs["sku_ids"]]
        costs = np.asarray(obs["costs"], dtype=float)
        inventory = np.asarray(obs["inventory"], dtype=float)
        precos = {
            offer_id: round(float(max(0.01, costs[idx] * (1.0 + self.markup))), 2)
            for idx, offer_id in enumerate(offer_ids)
        }
        pedidos = {
            sku_id: round(float(max(0.0, self.restock_target - inventory[idx])), 2)
            for idx, sku_id in enumerate(sku_ids)
        }
        markdowns = {offer_id: 0.0 for offer_id in offer_ids}
        return json.dumps(
            {"precos": precos, "pedidos": pedidos, "markdowns": markdowns}
        )


app = FastAPI(
    title="AVAL MVP",
    description="Agent Evaluation frontend API for economic rationality scenarios.",
    version="0.1.0",
)


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="web/index.html not found.")
    return FileResponse(index_path)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [
            _scenario_summary(scenario) for scenario in SCENARIOS.values()
        ]
    }


@app.get("/api/scenarios/{scenario_id}")
def scenario_detail(scenario_id: str) -> dict[str, Any]:
    try:
        scenario = get_scenario(scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _scenario_detail(scenario)


@app.post("/api/run")
def run_evaluation(request: RunRequest) -> dict[str, Any]:
    _validate_seeds(request.seeds)
    scenario_ids = _validate_scenario_ids(request.scenario_ids)
    agent = _agent_from_name(request.agent)
    batch = run_multi_seed(
        agent=agent,
        seeds=request.seeds,
        scenario_ids=scenario_ids,
        stochastic=request.stochastic,
        scenario_rng_seed=request.scenario_rng_seed,
        report_path=str(REPORT_PATH) if request.generate_report else None,
    )
    return _json_safe(batch)


@app.post("/api/manual-run")
def run_manual(request: ManualRunRequest) -> dict[str, Any]:
    try:
        scenario = get_scenario(request.scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    _validate_manual_action(scenario, request.prices, request.orders, request.markdowns)
    agent = FixedActionAgent(
        {
            "prices": request.prices,
            "orders": request.orders,
            "markdowns": request.markdowns,
        }
    )
    batch = run_multi_seed(
        agent=agent,
        seeds=[request.seed],
        scenario_ids=[scenario.scenario_id],
        stochastic=request.stochastic,
        scenario_rng_seed=0,
        report_path=str(REPORT_PATH) if request.generate_report else None,
    )
    return _json_safe(batch)


@app.post("/api/report")
def generate_report(request: ReportRequest) -> dict[str, Any]:
    if request.batch is None:
        batch = run_multi_seed(
            agent=NaiveAgent(),
            seeds=[101, 202, 303],
            stochastic=True,
            scenario_rng_seed=0,
        )
    else:
        batch = request.batch
    path = write_html_report(batch, REPORT_PATH)
    return {"report_path": path, "url": "/aval_parecer.html"}


@app.get("/aval_parecer.html")
def report_file() -> FileResponse:
    if not REPORT_PATH.exists():
        batch = run_multi_seed(
            agent=NaiveAgent(),
            seeds=[101, 202, 303],
            stochastic=True,
            scenario_rng_seed=0,
            report_path=str(REPORT_PATH),
        )
        if "report_path" not in batch:
            raise HTTPException(status_code=500, detail="Could not generate report.")
    return FileResponse(REPORT_PATH, media_type="text/html")


def _agent_from_name(name: str):
    if name == "naive":
        return NaiveAgent()
    if name == "oracle":
        return OracleAgent()
    if name == "fixed_markup":
        return FixedMarkupAgent()
    if name == "random":
        return RandomAgent(seed=0)
    if name in ("heuristic_llm", "mock_llm"):
        return HeuristicLLMAgent()
    raise HTTPException(status_code=400, detail=f"Unknown agent: {name}.")


def _scenario_summary(scenario: ScenarioSpec) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "label": scenario.label,
        "engine": scenario.engine,
        "horizon_days": scenario.horizon_days,
        "sku_count": len(scenario.skus),
        "bundle_count": len(scenario.bundles),
        "capacity_units": scenario.capacity_units,
        "metadata": dict(scenario.metadata),
    }


def _scenario_detail(scenario: ScenarioSpec) -> dict[str, Any]:
    matrices = scenario.to_matrices(include_bundles=True)
    return {
        **_scenario_summary(scenario),
        "offer_ids": [str(item) for item in matrices["offer_ids"]],
        "sku_ids": [sku.sku_id for sku in scenario.skus],
        "bundle_ids": [bundle.bundle_id for bundle in scenario.bundles],
        "costs": _float_list(matrices["costs"]),
        "initial_prices": _float_list(matrices["initial_prices"]),
        "shelf_life_days": _float_list(matrices["shelf_life_days"]),
        "salvage_values": _float_list(matrices["salvage_values"]),
        "max_order_units": _float_list(matrices["max_order_units"]),
        "demand_params": [
            _float_list(row) for row in np.asarray(matrices["demand_params"])
        ],
        "demand_param_columns": list(matrices["demand_param_columns"]),
        "is_bundle": [bool(item) for item in matrices["is_bundle"]],
        "bundle_component_matrix": [
            _float_list(row)
            for row in np.asarray(matrices["bundle_component_matrix"], dtype=float)
        ],
    }


def _validate_scenario_ids(scenario_ids: list[str] | None) -> list[str] | None:
    if scenario_ids is None:
        return None
    if not scenario_ids:
        raise HTTPException(status_code=400, detail="scenario_ids cannot be empty.")
    if len(scenario_ids) > MAX_SCENARIOS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many scenario_ids (max {MAX_SCENARIOS}).",
        )
    for scenario_id in scenario_ids:
        try:
            get_scenario(scenario_id)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    return scenario_ids


def _validate_seeds(seeds: list[int]) -> None:
    if not seeds:
        raise HTTPException(status_code=400, detail="seeds cannot be empty.")
    if len(seeds) > MAX_SEEDS:
        raise HTTPException(
            status_code=400,
            detail=f"Too many seeds (max {MAX_SEEDS}).",
        )


def _validate_manual_action(
    scenario: ScenarioSpec,
    prices: Mapping[str, float],
    orders: Mapping[str, float],
    markdowns: Mapping[str, float],
) -> None:
    offer_ids = set(scenario.offer_ids(include_bundles=True))
    sku_ids = {sku.sku_id for sku in scenario.skus}
    missing_prices = offer_ids - set(prices)
    missing_orders = sku_ids - set(orders)
    unknown_prices = set(prices) - offer_ids
    unknown_orders = set(orders) - sku_ids
    unknown_markdowns = set(markdowns) - offer_ids
    if missing_prices:
        raise HTTPException(
            status_code=400,
            detail=f"Missing prices: {', '.join(sorted(missing_prices))}.",
        )
    if missing_orders:
        raise HTTPException(
            status_code=400,
            detail=f"Missing orders: {', '.join(sorted(missing_orders))}.",
        )
    if unknown_prices or unknown_orders or unknown_markdowns:
        unknown = sorted(unknown_prices | unknown_orders | unknown_markdowns)
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ids: {', '.join(unknown)}.",
        )
    if any(float(value) <= 0.0 for value in prices.values()):
        raise HTTPException(status_code=400, detail="All prices must be positive.")
    if any(float(value) < 0.0 for value in orders.values()):
        raise HTTPException(status_code=400, detail="Orders must be non-negative.")
    if any(float(value) < 0.0 or float(value) >= 1.0 for value in markdowns.values()):
        raise HTTPException(status_code=400, detail="Markdowns must be in [0, 1).")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return value
    return value


def _float_list(value: Any) -> list[float]:
    return [float(item) for item in np.asarray(value, dtype=float).reshape(-1)]
