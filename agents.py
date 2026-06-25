"""Agent interfaces for AVAL simulations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from typing import Any, Protocol

import numpy as np


DEFAULT_SYSTEM_PROMPT = (
    "You are an autonomous retail pricing and inventory agent. "
    "Choose economically coherent prices, SKU orders, and markdowns. "
    "Return JSON only, with no prose."
)


class AgentParseError(ValueError):
    """Raised when an LLM response cannot be converted into an environment action."""


class ModelClientNotConfigured(RuntimeError):
    """Raised when LLMAgent.act is called without a completion client."""


@dataclass(frozen=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    model: str
    temperature: float = 0.0
    response_format: str = "json_object"

    @property
    def messages(self) -> tuple[dict[str, str], dict[str, str]]:
        return (
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        )


class CompletionClient(Protocol):
    def complete(self, request: LLMRequest) -> str:
        """Return model text for a provider-neutral completion request."""


@dataclass(frozen=True)
class AgentDecision:
    precos: dict[str, float]
    pedidos: dict[str, float]
    markdowns: dict[str, float]
    raw_text: str = ""

    def to_env_action(self) -> dict[str, dict[str, float]]:
        return {
            "prices": dict(self.precos),
            "orders": dict(self.pedidos),
            "markdowns": dict(self.markdowns),
        }


class LLMAgent:
    """Provider-neutral LLM agent.

    The client can be either:
    - an object with complete(request: LLMRequest) -> str
    - a callable with the same request object -> str
    """

    def __init__(
        self,
        client: CompletionClient | Callable[[LLMRequest], str] | None = None,
        model: str = "frontier-model",
        temperature: float = 0.0,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = float(temperature)
        self.system_prompt = system_prompt

    def act(self, obs: Mapping[str, Any]) -> dict[str, dict[str, float]]:
        decision = self.decide(obs)
        return decision.to_env_action()

    def decide(self, obs: Mapping[str, Any]) -> AgentDecision:
        prompt = self._render(obs)
        raw_text = self._call_model(prompt)
        return self.parse(raw_text, obs)

    def render(self, obs: Mapping[str, Any]) -> str:
        return self._render(obs)

    def parse(self, text: str, obs: Mapping[str, Any]) -> AgentDecision:
        payload = self._load_json_payload(text)
        required = {"precos", "pedidos", "markdowns"}
        missing = required - set(payload)
        if missing:
            raise AgentParseError(
                "LLM response missing required keys: "
                + ", ".join(sorted(missing))
                + "."
            )

        offer_ids = _string_ids(obs, "offer_ids")
        sku_ids = _string_ids(obs, "sku_ids")

        precos = _coerce_named_vector(
            payload["precos"],
            offer_ids,
            "precos",
            require_complete=True,
            lower_bound=0.0,
            lower_inclusive=False,
        )
        pedidos = _coerce_named_vector(
            payload["pedidos"],
            sku_ids,
            "pedidos",
            require_complete=True,
            lower_bound=0.0,
            lower_inclusive=True,
        )
        markdowns = _coerce_named_vector(
            payload["markdowns"],
            offer_ids,
            "markdowns",
            require_complete=True,
            lower_bound=0.0,
            lower_inclusive=True,
            upper_bound=1.0,
            upper_inclusive=False,
        )
        return AgentDecision(
            precos=precos,
            pedidos=pedidos,
            markdowns=markdowns,
            raw_text=text,
        )

    def _render(self, obs: Mapping[str, Any]) -> str:
        offer_ids = _string_ids(obs, "offer_ids")
        sku_ids = _string_ids(obs, "sku_ids")
        bundle_ids = tuple(str(item) for item in obs.get("bundle_ids", ()))
        costs = _numeric_array(obs, "costs", len(offer_ids))
        current_prices = _numeric_array(obs, "current_prices", len(offer_ids))
        expected_demand = _numeric_array(obs, "expected_demand", len(offer_ids))
        intercepts = _numeric_array(obs, "demand_intercepts", len(offer_ids))
        slopes = _numeric_array(obs, "demand_slopes", len(offer_ids))
        inventory = _numeric_array(obs, "inventory", len(sku_ids))
        shelf_life = _numeric_array(obs, "shelf_life_days", len(sku_ids))

        lines = [
            "AVAL decision task",
            f"Scenario: {obs.get('scenario_id', 'unknown')}",
            f"Engine: {obs.get('engine', 'unknown')}",
            f"Day: {obs.get('day', '?')} of {obs.get('horizon_days', '?')}",
            "",
            "Offers",
        ]

        sku_count = len(sku_ids)
        for idx, offer_id in enumerate(offer_ids):
            kind = "bundle" if offer_id in bundle_ids else "sku"
            stock_text = "component constrained"
            life_text = "component minimum"
            if idx < sku_count:
                stock_text = _fmt(inventory[idx])
                life_text = f"{_fmt(shelf_life[idx])} days"
            lines.append(
                "- "
                f"{offer_id} | type={kind} | cost={_fmt(costs[idx])} | "
                f"current_price={_fmt(current_prices[idx])} | inventory={stock_text} | "
                f"shelf_life={life_text} | demand_a={_fmt(intercepts[idx])} | "
                f"demand_b={_fmt(slopes[idx])} | expected_demand_at_current_price="
                f"{_fmt(expected_demand[idx])}"
            )

        if bundle_ids:
            lines.extend(["", "Bundles"])
            component_matrix = np.asarray(
                obs.get("bundle_component_matrix", np.zeros((0, sku_count))),
                dtype=float,
            )
            for bundle_idx, bundle_id in enumerate(bundle_ids):
                components: list[str] = []
                if bundle_idx < component_matrix.shape[0]:
                    for sku_idx, qty in enumerate(component_matrix[bundle_idx]):
                        if qty > 0.0:
                            components.append(f"{_fmt(qty)} x {sku_ids[sku_idx]}")
                components_text = ", ".join(components) if components else "none"
                lines.append(f"- {bundle_id}: {components_text}")

        lines.extend(
            [
                "",
                "Return exactly one JSON object. Do not wrap it in markdown.",
                "Use these keys and include every id shown below:",
                _json_contract(offer_ids, sku_ids),
                "",
                "Constraints:",
                "- precos: positive numeric price for every offer id.",
                "- pedidos: non-negative numeric order quantity for every SKU id.",
                "- markdowns: numeric discount in [0, 1) for every offer id.",
            ]
        )
        return "\n".join(lines)

    def _call_model(self, prompt: str) -> str:
        if self.client is None:
            raise ModelClientNotConfigured(
                "LLMAgent requires a completion client before act() can be used."
            )
        request = LLMRequest(
            system_prompt=self.system_prompt,
            user_prompt=prompt,
            model=self.model,
            temperature=self.temperature,
        )

        complete = getattr(self.client, "complete", None)
        if callable(complete):
            raw_text = complete(request)
        elif callable(self.client):
            raw_text = self.client(request)
        else:
            raise TypeError("client must be callable or expose complete(request).")
        if not isinstance(raw_text, str):
            raise TypeError("completion client must return a string.")
        return raw_text

    @staticmethod
    def _load_json_payload(text: str) -> dict[str, Any]:
        if not isinstance(text, str) or not text.strip():
            raise AgentParseError("LLM response is empty.")
        json_text = _extract_json_object(text)
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise AgentParseError(f"Invalid JSON response: {exc.msg}.") from exc
        if not isinstance(payload, dict):
            raise AgentParseError("LLM response must be a JSON object.")
        return payload


def _json_contract(offer_ids: tuple[str, ...], sku_ids: tuple[str, ...]) -> str:
    contract = {
        "precos": {offer_id: "number" for offer_id in offer_ids},
        "pedidos": {sku_id: "number" for sku_id in sku_ids},
        "markdowns": {offer_id: "number" for offer_id in offer_ids},
    }
    return json.dumps(contract, indent=2, sort_keys=False)


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    if start < 0:
        raise AgentParseError("LLM response does not contain a JSON object.")

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(stripped)):
        char = stripped[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return stripped[start : idx + 1]
    raise AgentParseError("LLM response contains an incomplete JSON object.")


def _coerce_named_vector(
    value: Any,
    ids: tuple[str, ...],
    field_name: str,
    require_complete: bool,
    lower_bound: float | None = None,
    lower_inclusive: bool = True,
    upper_bound: float | None = None,
    upper_inclusive: bool = True,
) -> dict[str, float]:
    if isinstance(value, Mapping):
        unknown = set(str(key) for key in value) - set(ids)
        if unknown:
            raise AgentParseError(
                f"{field_name} contains unknown ids: {', '.join(sorted(unknown))}."
            )
        missing = set(ids) - set(str(key) for key in value)
        if require_complete and missing:
            raise AgentParseError(
                f"{field_name} missing ids: {', '.join(sorted(missing))}."
            )
        result = {item_id: 0.0 for item_id in ids}
        for key, raw_item in value.items():
            result[str(key)] = _coerce_number(raw_item, f"{field_name}.{key}")
    else:
        array = np.asarray(value, dtype=object)
        if array.shape != (len(ids),):
            raise AgentParseError(
                f"{field_name} must have {len(ids)} values, received {array.size}."
            )
        result = {
            item_id: _coerce_number(raw_item, f"{field_name}.{item_id}")
            for item_id, raw_item in zip(ids, array)
        }

    for item_id, number in result.items():
        _validate_bounds(
            number,
            f"{field_name}.{item_id}",
            lower_bound,
            lower_inclusive,
            upper_bound,
            upper_inclusive,
        )
    return result


def _coerce_number(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AgentParseError(f"{label} must be numeric.") from exc
    if not math.isfinite(number):
        raise AgentParseError(f"{label} must be finite.")
    return number


def _validate_bounds(
    value: float,
    label: str,
    lower_bound: float | None,
    lower_inclusive: bool,
    upper_bound: float | None,
    upper_inclusive: bool,
) -> None:
    if lower_bound is not None:
        if lower_inclusive and value < lower_bound:
            raise AgentParseError(f"{label} must be >= {lower_bound}.")
        if not lower_inclusive and value <= lower_bound:
            raise AgentParseError(f"{label} must be > {lower_bound}.")
    if upper_bound is not None:
        if upper_inclusive and value > upper_bound:
            raise AgentParseError(f"{label} must be <= {upper_bound}.")
        if not upper_inclusive and value >= upper_bound:
            raise AgentParseError(f"{label} must be < {upper_bound}.")


def _string_ids(obs: Mapping[str, Any], key: str) -> tuple[str, ...]:
    if key not in obs:
        raise KeyError(f"Observation missing {key}.")
    return tuple(str(item) for item in obs[key])


def _numeric_array(obs: Mapping[str, Any], key: str, expected_size: int) -> np.ndarray:
    if key not in obs:
        raise KeyError(f"Observation missing {key}.")
    array = np.asarray(obs[key], dtype=float)
    if array.shape != (expected_size,):
        raise ValueError(f"Observation {key} must have {expected_size} values.")
    return array


def _fmt(value: float) -> str:
    return f"{float(value):.2f}"

