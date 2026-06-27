const state = {
  scenarios: [],
  details: new Map(),
  batch: null,
};

const els = {
  agentSelect: document.getElementById("agentSelect"),
  scenarioChecks: document.getElementById("scenarioChecks"),
  seedInput: document.getElementById("seedInput"),
  stochasticInput: document.getElementById("stochasticInput"),
  runButton: document.getElementById("runButton"),
  reportButton: document.getElementById("reportButton"),
  statusText: document.getElementById("statusText"),
  manualPanel: document.getElementById("manualPanel"),
  manualScenarioSelect: document.getElementById("manualScenarioSelect"),
  manualEditor: document.getElementById("manualEditor"),
  heroDistance: document.getElementById("heroDistance"),
  heroScore: document.getElementById("heroScore"),
  heroGrade: document.getElementById("heroGrade"),
  heroVerdict: document.getElementById("heroVerdict"),
  gaugeFill: document.getElementById("gaugeFill"),
  gaugeProximity: document.getElementById("gaugeProximity"),
  pillarPriceGap: document.getElementById("pillarPriceGap"),
  pillarOrderGap: document.getElementById("pillarOrderGap"),
  pillarEfficiency: document.getElementById("pillarEfficiency"),
  kpiSurvival: document.getElementById("kpiSurvival"),
  kpiStructural: document.getElementById("kpiStructural"),
  kpiProfit: document.getElementById("kpiProfit"),
  batchMeta: document.getElementById("batchMeta"),
  diagnosticGrid: document.getElementById("diagnosticGrid"),
  episodeRows: document.getElementById("episodeRows"),
  episodeSelect: document.getElementById("episodeSelect"),
  stepMeta: document.getElementById("stepMeta"),
  stepRows: document.getElementById("stepRows"),
  reportLink: document.getElementById("reportLink"),
};

const scenarioNames = {
  buybye_autonomous: "BuyBye - varejo autonomo",
  d2c_fashion: "WearClio/Disturb - moda D2C",
  baco_premium: "Seja Baco - bebidas premium",
};

// A relative price gap at or above this magnitude fills a per-offer distance bar
// completely. 50% off the oracle price is already a gross miss.
const PRICE_GAP_FULL_SCALE = 0.5;

async function init() {
  bindEvents();
  await loadScenarios();
  updateManualVisibility();
  await loadManualScenario();
  setStatus("Catalogo carregado. Rode uma avaliacao.");
}

function bindEvents() {
  els.agentSelect.addEventListener("change", updateManualVisibility);
  els.manualScenarioSelect.addEventListener("change", loadManualScenario);
  els.runButton.addEventListener("click", runEvaluation);
  els.reportButton.addEventListener("click", generateReport);
  els.episodeSelect.addEventListener("change", renderSelectedEpisode);
}

async function loadScenarios() {
  const data = await fetchJson("/api/scenarios");
  state.scenarios = data.scenarios;
  els.scenarioChecks.innerHTML = "";
  els.manualScenarioSelect.innerHTML = "";

  for (const scenario of state.scenarios) {
    const label = document.createElement("label");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.value = scenario.scenario_id;
    input.checked = true;
    label.append(input, document.createTextNode(labelForScenario(scenario.scenario_id)));
    els.scenarioChecks.appendChild(label);

    const option = document.createElement("option");
    option.value = scenario.scenario_id;
    option.textContent = labelForScenario(scenario.scenario_id);
    els.manualScenarioSelect.appendChild(option);
  }
}

async function loadScenarioDetail(scenarioId) {
  if (!state.details.has(scenarioId)) {
    const detail = await fetchJson(`/api/scenarios/${encodeURIComponent(scenarioId)}`);
    state.details.set(scenarioId, detail);
  }
  return state.details.get(scenarioId);
}

async function loadManualScenario() {
  const scenarioId = els.manualScenarioSelect.value || state.scenarios[0]?.scenario_id;
  if (!scenarioId) return;
  const detail = await loadScenarioDetail(scenarioId);
  renderManualEditor(detail);
}

function renderManualEditor(detail) {
  const offerIds = detail.offer_ids;
  const skuIds = detail.sku_ids;
  const priceById = new Map(offerIds.map((id, index) => [id, detail.initial_prices[index]]));

  els.manualEditor.innerHTML = "";
  const group = document.createElement("div");
  group.className = "manual-group";
  group.innerHTML = `
    <h3>${escapeHtml(labelForScenario(detail.scenario_id))}</h3>
    <div class="manual-row manual-header">
      <span>ID</span><span>Preco</span><span>Pedido</span><span>Markdown</span>
    </div>
  `;

  for (const offerId of offerIds) {
    const isSku = skuIds.includes(offerId);
    const row = document.createElement("div");
    row.className = "manual-row";
    row.innerHTML = `
      <span title="${escapeHtml(offerId)}">${escapeHtml(offerId)}</span>
      <input type="number" min="0.01" step="0.01" data-kind="price" data-id="${escapeHtml(offerId)}" value="${formatInput(priceById.get(offerId))}">
      <input type="number" min="0" step="1" data-kind="order" data-id="${escapeHtml(offerId)}" value="${isSku ? "1" : "0"}" ${isSku ? "" : "disabled"}>
      <input type="number" min="0" max="0.99" step="0.01" data-kind="markdown" data-id="${escapeHtml(offerId)}" value="0">
    `;
    group.appendChild(row);
  }
  els.manualEditor.appendChild(group);
}

function updateManualVisibility() {
  const manual = els.agentSelect.value === "manual";
  els.manualPanel.classList.toggle("hidden", !manual);
}

async function runEvaluation() {
  setBusy(true);
  try {
    const agent = els.agentSelect.value;
    const seeds = parseSeeds();
    let batch;
    if (agent === "manual") {
      const scenarioId = els.manualScenarioSelect.value;
      const action = readManualAction();
      batch = await fetchJson("/api/manual-run", {
        method: "POST",
        body: JSON.stringify({
          scenario_id: scenarioId,
          seed: seeds[0],
          stochastic: els.stochasticInput.checked,
          ...action,
        }),
      });
    } else {
      const scenarioIds = selectedScenarioIds();
      batch = await fetchJson("/api/run", {
        method: "POST",
        body: JSON.stringify({
          agent,
          scenario_ids: scenarioIds,
          seeds,
          stochastic: els.stochasticInput.checked,
          scenario_rng_seed: 0,
        }),
      });
    }
    state.batch = batch;
    renderBatch(batch);
    setStatus("Avaliacao concluida.");
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    setBusy(false);
  }
}

async function generateReport() {
  setBusy(true);
  try {
    const payload = state.batch ? { batch: state.batch } : {};
    const data = await fetchJson("/api/report", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    els.reportLink.href = data.url;
    setStatus("Parecer gerado. Use o link no topo para abrir.");
    window.open(data.url, "_blank", "noreferrer");
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    setBusy(false);
  }
}

function renderBatch(batch) {
  const summary = batch.summary || {};
  renderHero(summary);

  setKpi(els.kpiSurvival, fmtPct(summary.survival_rate), efficiencyClass(summary.survival_rate));
  setKpi(els.kpiStructural, fmtPct(summary.mean_structural_r_squared), efficiencyClass(summary.mean_structural_r_squared));
  els.kpiProfit.textContent = fmtMoney(summary.total_profit);
  els.batchMeta.textContent = `${batch.agent_name} | ${batch.episodes.length} episodios | ${batch.seeds.join(", ")}`;

  renderDiagnostics(batch);
  renderEpisodeRows(batch);
  populateEpisodeSelect(batch);
  renderSelectedEpisode();
}

function renderHero(summary) {
  const distance = distanceToOptimal(summary.mean_price_efficiency);
  const proximity = proximityToOptimal(summary.mean_price_efficiency);
  const distClass = distanceClass(distance);

  setKpi(els.heroDistance, fmtPct(distance), distClass);

  els.gaugeFill.style.width = `${(Number.isFinite(proximity) ? proximity : 0) * 100}%`;
  els.gaugeFill.classList.remove("good", "warn", "bad");
  if (distClass) els.gaugeFill.classList.add(distClass);
  els.gaugeProximity.textContent = fmtPct(proximity);

  const grade = summary.grade ? `${summary.grade}` : "";
  els.heroScore.textContent = fmtNum(summary.mean_aval_score);
  els.heroScore.classList.remove("good", "warn", "bad");
  const sClass = scoreClass(summary.mean_aval_score);
  if (sClass) els.heroScore.classList.add(sClass);
  els.heroGrade.textContent = grade;
  setKpi(els.heroVerdict, summary.verdict || "-", verdictClass(summary.verdict));

  setKpi(els.pillarPriceGap, fmtPct(summary.mean_abs_relative_price_gap), gapClass(summary.mean_abs_relative_price_gap));
  setKpi(els.pillarOrderGap, fmtNum(summary.mean_abs_order_gap), orderGapClass(summary.mean_abs_order_gap));
  setKpi(els.pillarEfficiency, fmtPct(summary.mean_price_efficiency), efficiencyClass(summary.mean_price_efficiency));
}

function renderDiagnostics(batch) {
  const scenarios = batch.summary?.scenarios || {};
  const cards = [];
  for (const [scenarioId, item] of Object.entries(scenarios).sort()) {
    const diagnostic = selectDiagnostic(item.diagnostics || []);
    cards.push(`
      <article class="diagnostic">
        <h3>${escapeHtml(labelForScenario(scenarioId))}</h3>
        ${badge(diagnostic.status)}
        <p>${escapeHtml(diagnostic.message || "Sem diagnostico.")}</p>
        <p class="muted">Modo: ${escapeHtml(diagnostic.failure_mode || "n/a")}</p>
      </article>
    `);
  }
  els.diagnosticGrid.innerHTML = cards.join("") || "<p>Sem diagnosticos.</p>";
}

function renderEpisodeRows(batch) {
  const rows = batch.episodes.map((episode) => {
    const summary = episode.summary;
    const diagnostic = summary.diagnostic || {};
    const distance = distanceToOptimal(summary.mean_price_efficiency);
    return `
      <tr>
        <td>${episode.seed}</td>
        <td>${escapeHtml(labelForScenario(episode.scenario_id))}</td>
        <td>${summary.days_run}</td>
        <td class="lead-col"><span class="dist-tag ${distanceClass(distance)}">${fmtPct(distance)}</span></td>
        <td>${fmtNum(summary.aval_score)}</td>
        <td>${escapeHtml(summary.grade || "-")}</td>
        <td>${fmtPct(summary.mean_abs_relative_price_gap)}</td>
        <td>${fmtNum(summary.mean_abs_order_gap)}</td>
        <td>${fmtMoney(summary.total_profit)}</td>
        <td>${badge(diagnostic.status)} ${escapeHtml(diagnostic.failure_mode || "")}</td>
      </tr>
    `;
  });
  els.episodeRows.innerHTML = rows.join("");
}

function populateEpisodeSelect(batch) {
  els.episodeSelect.innerHTML = "";
  batch.episodes.forEach((episode, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `${episode.seed} | ${labelForScenario(episode.scenario_id)}`;
    els.episodeSelect.appendChild(option);
  });
}

function renderSelectedEpisode() {
  if (!state.batch || !state.batch.episodes.length) {
    els.stepRows.innerHTML = '<tr><td colspan="10">Sem dados.</td></tr>';
    return;
  }
  const index = Number(els.episodeSelect.value || 0);
  const episode = state.batch.episodes[index];
  els.stepMeta.textContent = `${labelForScenario(episode.scenario_id)} | seed ${episode.seed}`;
  const rows = [];
  for (const step of episode.steps) {
    const skuIds = step.sku_ids || [];
    const offerIds = step.offer_ids || [];
    offerIds.forEach((offerId, offerIndex) => {
      const skuIndex = skuIds.indexOf(offerId);
      rows.push(`
        <tr>
          <td>${step.day}</td>
          <td>${escapeHtml(offerId)}</td>
          <td>${fmtMoney(step.prices[offerIndex])}</td>
          <td>${fmtMoney(step.oracle_prices[offerIndex])}</td>
          <td class="lead-col">${distanceBar(step.relative_price_gap[offerIndex])}</td>
          <td>${skuIndex >= 0 ? fmtNum(step.orders[skuIndex]) : "-"}</td>
          <td>${skuIndex >= 0 ? fmtNum(step.oracle_orders[skuIndex]) : "-"}</td>
          <td>${fmtNum(step.lost_sales[offerIndex])}</td>
          <td>${skuIndex >= 0 ? fmtNum(step.expired_units[skuIndex]) : "-"}</td>
          <td>${offerIndex >= skuIds.length ? fmtNum(step.bundle_sales[offerIndex - skuIds.length]) : "-"}</td>
        </tr>
      `);
    });
  }
  els.stepRows.innerHTML = rows.join("") || '<tr><td colspan="10">Sem dados.</td></tr>';
}

function selectedScenarioIds() {
  const ids = [...els.scenarioChecks.querySelectorAll("input:checked")].map((input) => input.value);
  if (!ids.length) throw new Error("Selecione pelo menos um cenario.");
  return ids;
}

function parseSeeds() {
  const seeds = els.seedInput.value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean)
    .map((item) => Number(item));
  if (!seeds.length || seeds.some((seed) => !Number.isInteger(seed))) {
    throw new Error("Informe seeds inteiras separadas por virgula.");
  }
  return seeds;
}

function readManualAction() {
  const prices = {};
  const orders = {};
  const markdowns = {};
  const inputs = els.manualEditor.querySelectorAll("input[data-kind]");
  inputs.forEach((input) => {
    const id = input.dataset.id;
    const value = Number(input.value);
    if (input.dataset.kind === "price") prices[id] = value;
    if (input.dataset.kind === "order" && !input.disabled) orders[id] = value;
    if (input.dataset.kind === "markdown") markdowns[id] = value;
  });
  return { prices, orders, markdowns };
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const data = await response.json();
      detail = data.detail || JSON.stringify(data);
    } catch (_) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function selectDiagnostic(diagnostics) {
  return diagnostics.find((item) => item.status === "fail") || diagnostics[0] || {};
}

// Distance to the theoretical optimum, derived from profit efficiency in [0, 1].
// efficiency = agent profit / oracle profit, so 1 - efficiency is the share of the
// optimum left on the table. Clamped at 0 because in-sample noise can push a single
// run marginally past the oracle.
function distanceToOptimal(efficiency) {
  const number = Number(efficiency);
  if (!Number.isFinite(number)) return NaN;
  return Math.max(0, 1 - number);
}

function proximityToOptimal(efficiency) {
  const number = Number(efficiency);
  if (!Number.isFinite(number)) return NaN;
  return Math.min(1, Math.max(0, number));
}

function distanceBar(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const magnitude = Math.abs(number);
  const fill = Math.min(1, magnitude / PRICE_GAP_FULL_SCALE) * 100;
  const cls = gapClass(magnitude);
  const sign = number > 0 ? "+" : "";
  return `
    <span class="dist-cell">
      <span class="dist-bar"><span class="dist-bar-fill ${cls}" style="width:${fill}%"></span></span>
      <span class="dist-bar-label ${cls}">${sign}${(number * 100).toFixed(1)}%</span>
    </span>
  `;
}

function setKpi(element, text, className) {
  element.textContent = text;
  element.classList.remove("good", "warn", "bad");
  if (className) element.classList.add(className);
}

function efficiencyClass(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number >= 0.95) return "good";
  if (number >= 0.8) return "warn";
  return "bad";
}

// Mirror of efficiencyClass for the distance framing: small distance is good.
function distanceClass(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number <= 0.05) return "good";
  if (number <= 0.2) return "warn";
  return "bad";
}

function scoreClass(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "";
  if (number >= 80) return "good";
  if (number >= 60) return "warn";
  return "bad";
}

function verdictClass(verdict) {
  if (verdict === "APROVADO") return "good";
  if (verdict === "RESSALVA") return "warn";
  if (verdict === "REPROVADO") return "bad";
  return "";
}

function gapClass(value) {
  const number = Math.abs(Number(value));
  if (!Number.isFinite(number)) return "";
  if (number <= 0.05) return "good";
  if (number <= 0.15) return "warn";
  return "bad";
}

function orderGapClass(value) {
  const number = Math.abs(Number(value));
  if (!Number.isFinite(number)) return "";
  if (number <= 5) return "good";
  if (number <= 15) return "warn";
  return "bad";
}

function badge(status) {
  const value = ["pass", "fail", "unknown"].includes(status) ? status : "unknown";
  const label = value === "pass" ? "OK" : value === "fail" ? "FALHA" : "N/A";
  return `<span class="badge ${value}">${label}</span>`;
}

function labelForScenario(id) {
  return scenarioNames[id] || id;
}

function fmtPct(value) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(1)}%` : "-";
}

function fmtNum(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "-";
}

function fmtMoney(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toLocaleString("en-US", { maximumFractionDigits: 2, minimumFractionDigits: 2 }) : "-";
}

function formatInput(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(2) : "0";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setBusy(busy) {
  els.runButton.disabled = busy;
  els.reportButton.disabled = busy;
}

function setStatus(message, isError = false) {
  els.statusText.textContent = message;
  els.statusText.style.color = isError ? "var(--bad)" : "var(--muted)";
}

init().catch((error) => {
  setStatus(error.message || String(error), true);
});
