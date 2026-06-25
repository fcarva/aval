"""AVAL Score composition: five economic layers -> 0-100 -> grade -> verdict.

This is the certification artifact the platform exists to produce. Each layer is
normalized to [0, 1] from the episode summary, combined with fixed weights into a
0-100 AVAL Score, and mapped to a standardized grade (A+..F) and an operational
verdict (APROVADO / RESSALVA / REPROVADO).
"""

from __future__ import annotations

from typing import Any, Mapping

import math


# Layer weights. Pricing and inventory dominate because they capture the bulk of
# realized economic value; structural recovery certifies the agent runs a sane
# model; flags and coherence are guardrails against ruinous behavior.
LAYER_WEIGHTS = {
    "pricing": 0.30,
    "inventory": 0.25,
    "structural": 0.20,
    "flags": 0.15,
    "coherence": 0.10,
}

GRADE_BANDS = (
    (95.0, "A+"),
    (90.0, "A"),
    (85.0, "A-"),
    (80.0, "B+"),
    (75.0, "B"),
    (70.0, "B-"),
    (65.0, "C+"),
    (60.0, "C"),
    (55.0, "C-"),
    (50.0, "D"),
)

APROVADO_THRESHOLD = 80.0
RESSALVA_THRESHOLD = 60.0


def _clip01(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return float(min(1.0, max(0.0, number)))


def layer_scores(summary: Mapping[str, Any]) -> dict[str, float]:
    """Normalize the raw episode summary into five [0, 1] layer scores."""

    pricing = _clip01(summary.get("mean_price_efficiency"))
    inventory = _clip01(summary.get("mean_order_accuracy"))
    structural = _clip01(summary.get("mean_structural_r_squared"))
    flags = _clip01(1.0 - _clip01(summary.get("flag_rate")))

    survived = 1.0 if summary.get("survived", True) else 0.0
    incoherence = _clip01(summary.get("price_incoherence"))
    coherence = 0.6 * survived + 0.4 * _clip01(1.0 - incoherence)

    return {
        "pricing": pricing,
        "inventory": inventory,
        "structural": structural,
        "flags": flags,
        "coherence": coherence,
    }


def grade_for_score(score: float) -> str:
    for threshold, grade in GRADE_BANDS:
        if score >= threshold:
            return grade
    return "F"


def verdict_for_score(score: float) -> str:
    if score >= APROVADO_THRESHOLD:
        return "APROVADO"
    if score >= RESSALVA_THRESHOLD:
        return "RESSALVA"
    return "REPROVADO"


def aval_certificate(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Produce the full certificate block for an episode summary."""

    layers = layer_scores(summary)
    score = 100.0 * sum(LAYER_WEIGHTS[name] * layers[name] for name in LAYER_WEIGHTS)
    score = round(score, 1)
    return {
        "aval_score": score,
        "grade": grade_for_score(score),
        "verdict": verdict_for_score(score),
        "layers": layers,
        "weights": dict(LAYER_WEIGHTS),
    }


def aggregate_certificate(scores: list[float]) -> dict[str, Any]:
    """Certificate for a batch: mean score -> grade -> verdict."""

    finite = [float(value) for value in scores if value is not None and math.isfinite(float(value))]
    if not finite:
        return {"aval_score": 0.0, "grade": "F", "verdict": "REPROVADO"}
    mean_score = round(sum(finite) / len(finite), 1)
    return {
        "aval_score": mean_score,
        "grade": grade_for_score(mean_score),
        "verdict": verdict_for_score(mean_score),
    }
