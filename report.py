"""HTML report rendering for AVAL simulation batches."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_REPORT_PATH = "aval_parecer.html"


def write_html_report(
    batch: Mapping[str, Any], output_path: str | Path = DEFAULT_REPORT_PATH
) -> str:
    path = Path(output_path)
    html = render_html_report(batch)
    path.write_text(html, encoding="utf-8")
    return str(path)


def render_html_report(batch: Mapping[str, Any]) -> str:
    summary = batch.get("summary", {})
    episodes = list(batch.get("episodes", []))
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scenario_cards = _render_scenario_cards(summary.get("scenarios", {}))
    episode_rows = _render_episode_rows(episodes)
    structural_rows = _render_structural_diagnostics(episodes)
    raw_json = escape(json.dumps(_json_safe(batch), ensure_ascii=True, indent=2))

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AVAL Parecer</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --ink: #1f2933;
      --muted: #627083;
      --line: #d9dee7;
      --panel: #ffffff;
      --good: #0f766e;
      --bad: #b42318;
      --warn: #9a6700;
      --accent: #244c88;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 Arial, Helvetica, sans-serif;
    }}
    header {{
      background: #172033;
      color: white;
      padding: 28px 32px;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 28px 24px 48px;
    }}
    h1, h2, h3 {{
      margin: 0;
      letter-spacing: 0;
    }}
    h1 {{
      font-size: 28px;
      font-weight: 700;
    }}
    h2 {{
      font-size: 20px;
      margin: 30px 0 12px;
    }}
    h3 {{
      font-size: 15px;
      margin-bottom: 8px;
    }}
    .subtitle {{
      margin-top: 8px;
      color: #cbd5e1;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
    }}
    .metric, .scenario {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
    }}
    .metric-label, .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    .metric-value {{
      font-size: 24px;
      font-weight: 700;
      margin-top: 4px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 9px 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eef2f7;
      color: #364152;
      font-size: 12px;
      text-transform: uppercase;
    }}
    tr:last-child td {{
      border-bottom: 0;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }}
    .pass {{
      background: #d7f3ee;
      color: var(--good);
    }}
    .fail {{
      background: #fee4e2;
      color: var(--bad);
    }}
    .unknown {{
      background: #fff2cc;
      color: var(--warn);
    }}
    .scenario p {{
      margin: 6px 0 0;
    }}
    details {{
      margin-top: 20px;
    }}
    pre {{
      max-height: 320px;
      overflow: auto;
      background: #111827;
      color: #e5e7eb;
      padding: 12px;
      border-radius: 8px;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>AVAL Parecer</h1>
    <div class="subtitle">Agent Evaluation - distancia para o otimo teorico e recuperacao estrutural. Gerado em {generated_at}.</div>
  </header>
  <main>
    <section class="grid" aria-label="Resumo executivo">
      {_metric_card("Episodios", summary.get("episodes", len(episodes)))}
      {_metric_card("Eficiencia media de preco", _fmt_pct(summary.get("mean_price_efficiency")))}
      {_metric_card("Gap medio de preco", _fmt_pct(summary.get("mean_abs_relative_price_gap")))}
      {_metric_card("Gap medio de pedido", _fmt_num(summary.get("mean_abs_order_gap")))}
      {_metric_card("Lucro simulado", _fmt_money(summary.get("total_profit")))}
    </section>

    <h2>Diagn&oacute;stico estrutural</h2>
    <section class="grid">
      {structural_rows}
    </section>

    <h2>Resumo por cen&aacute;rio</h2>
    <table>
      <thead>
        <tr>
          <th>Cen&aacute;rio</th>
          <th>Epis&oacute;dios</th>
          <th>Efici&ecirc;ncia de pre&ccedil;o</th>
          <th>Gap de pre&ccedil;o</th>
          <th>Gap de pedido</th>
          <th>Lucro</th>
          <th>Diagn&oacute;sticos</th>
        </tr>
      </thead>
      <tbody>
        {scenario_cards}
      </tbody>
    </table>

    <h2>Epis&oacute;dios</h2>
    <table>
      <thead>
        <tr>
          <th>Seed</th>
          <th>Cen&aacute;rio</th>
          <th>Dias</th>
          <th>Efici&ecirc;ncia</th>
          <th>Gap pre&ccedil;o</th>
          <th>Gap pedido</th>
          <th>Lucro</th>
          <th>Diagn&oacute;stico</th>
        </tr>
      </thead>
      <tbody>
        {episode_rows}
      </tbody>
    </table>

    <details>
      <summary>Dados estruturados</summary>
      <pre>{raw_json}</pre>
    </details>
  </main>
</body>
</html>
"""


def _metric_card(label: str, value: Any) -> str:
    return (
        '<div class="metric">'
        f'<div class="metric-label">{escape(str(label))}</div>'
        f'<div class="metric-value">{escape(str(value))}</div>'
        "</div>"
    )


def _render_structural_diagnostics(episodes: Sequence[Mapping[str, Any]]) -> str:
    by_scenario: dict[str, list[Mapping[str, Any]]] = {}
    for episode in episodes:
        summary = episode.get("summary", {})
        diagnostic = summary.get("diagnostic", {})
        scenario_id = str(summary.get("scenario_id", episode.get("scenario_id", "")))
        by_scenario.setdefault(scenario_id, []).append(diagnostic)

    if not by_scenario:
        return '<div class="scenario">Sem episodios avaliados.</div>'

    cards: list[str] = []
    for scenario_id in sorted(by_scenario):
        diagnostics = by_scenario[scenario_id]
        selected = _select_representative_diagnostic(diagnostics)
        cards.append(
            '<article class="scenario">'
            f"<h3>{escape(_scenario_name(scenario_id))}</h3>"
            f"{_badge(selected.get('status', 'unknown'))}"
            f"<p>{escape(str(selected.get('message', 'Sem diagnostico.')))}</p>"
            f'<p class="muted">Modo: {escape(str(selected.get("failure_mode", "")))}</p>'
            "</article>"
        )
    return "\n".join(cards)


def _render_scenario_cards(scenarios: Mapping[str, Mapping[str, Any]]) -> str:
    if not scenarios:
        return '<tr><td colspan="7">Sem cenarios avaliados.</td></tr>'
    rows: list[str] = []
    for scenario_id, summary in sorted(scenarios.items()):
        diagnostics = summary.get("diagnostics", [])
        diagnostic_text = "; ".join(
            sorted(
                {
                    str(item.get("failure_mode", "nao classificado"))
                    for item in diagnostics
                    if isinstance(item, Mapping)
                }
            )
        )
        rows.append(
            "<tr>"
            f"<td>{escape(_scenario_name(scenario_id))}</td>"
            f"<td>{escape(str(summary.get('episodes', 0)))}</td>"
            f"<td>{escape(_fmt_pct(summary.get('mean_price_efficiency')))}</td>"
            f"<td>{escape(_fmt_pct(summary.get('mean_abs_relative_price_gap')))}</td>"
            f"<td>{escape(_fmt_num(summary.get('mean_abs_order_gap')))}</td>"
            f"<td>{escape(_fmt_money(summary.get('total_profit')))}</td>"
            f"<td>{escape(diagnostic_text or 'sem diagnostico')}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _render_episode_rows(episodes: Sequence[Mapping[str, Any]]) -> str:
    if not episodes:
        return '<tr><td colspan="8">Sem episodios avaliados.</td></tr>'
    rows: list[str] = []
    for episode in episodes:
        summary = episode.get("summary", {})
        diagnostic = summary.get("diagnostic", {})
        rows.append(
            "<tr>"
            f"<td>{escape(str(episode.get('seed', '')))}</td>"
            f"<td>{escape(_scenario_name(str(episode.get('scenario_id', ''))))}</td>"
            f"<td>{escape(str(summary.get('days_run', '')))}</td>"
            f"<td>{escape(_fmt_pct(summary.get('mean_price_efficiency')))}</td>"
            f"<td>{escape(_fmt_pct(summary.get('mean_abs_relative_price_gap')))}</td>"
            f"<td>{escape(_fmt_num(summary.get('mean_abs_order_gap')))}</td>"
            f"<td>{escape(_fmt_money(summary.get('total_profit')))}</td>"
            f"<td>{_badge(diagnostic.get('status', 'unknown'))} {escape(str(diagnostic.get('message', '')))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def _select_representative_diagnostic(
    diagnostics: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    for diagnostic in diagnostics:
        if diagnostic.get("status") == "fail":
            return diagnostic
    return diagnostics[0] if diagnostics else {}


def _badge(status: Any) -> str:
    value = str(status or "unknown")
    css_class = value if value in {"pass", "fail", "unknown"} else "unknown"
    label = {"pass": "OK", "fail": "FALHA", "unknown": "N/A"}.get(css_class, "N/A")
    return f'<span class="badge {css_class}">{label}</span>'


def _scenario_name(scenario_id: str) -> str:
    return {
        "buybye_autonomous": "BuyBye - varejo autonomo",
        "d2c_fashion": "WearClio/Disturb - moda D2C",
        "baco_premium": "Seja Baco - bebidas premium",
    }.get(scenario_id, scenario_id)


def _fmt_pct(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.1f}%"


def _fmt_num(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:.2f}"


def _fmt_money(value: Any) -> str:
    number = _to_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.2f}"


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if isinstance(value, float):
        if value != value:
            return None
        return value
    return value

