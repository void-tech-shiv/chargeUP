const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));

function showPage(name) {
  $$(".page").forEach(p => p.classList.toggle("active", p.id === `page-${name}`));
  $$(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.page === name));
  $("#sidebar")?.classList.remove("open");
  window.scrollTo({top: 0, behavior: "smooth"});
  if (name === "monitoring") loadMetrics();
  if (name === "overview") loadStations();
  if (name === "clusters") loadClusters();
  if (name === "performance") loadFeatureImportance();
}

$$(".nav-item").forEach(btn => btn.addEventListener("click", () => showPage(btn.dataset.page)));
$("#menuBtn")?.addEventListener("click", () => $("#sidebar")?.classList.toggle("open"));

function setDefaultTimestamp() {
  const input = $("#predictionTimestamp");
  if (!input || input.value) return;
  const now = new Date();
  const local = new Date(now.getTime() - now.getTimezoneOffset() * 60000);
  input.value = local.toISOString().slice(0, 16);
}

async function health() {
  try {
    const r = await fetch("/health");
    if (!r.ok) throw new Error("health failed");
    const j = await r.json();
    $("#health").textContent = `● API ONLINE • ${j.version}`;
    $("#health").className = "api-pill ok";
    if ($("#mobileHealth")) $("#mobileHealth").style.background = "var(--accent)";
  } catch {
    $("#health").textContent = "● API OFFLINE";
    $("#health").className = "api-pill fault";
    if ($("#mobileHealth")) $("#mobileHealth").style.background = "var(--danger)";
  }
}

function isRisk(label) {
  return label === "SERVER_TERMINATION_RISK" || label === "FAULT";
}

function badge(label) {
  const safe = esc(label);
  return `<span class="badge ${isRisk(label) ? "fault" : "ok"}">${safe}</span>`;
}

async function loadStations() {
  const body = $("#stationRows");
  if (!body) return;
  body.innerHTML = `<tr><td colspan="5">Loading real Palo Alto predictions…</td></tr>`;
  try {
    const r = await fetch("/demo/stations?limit=10");
    if (!r.ok) throw new Error("station request failed");
    const data = await r.json();
    body.innerHTML = data.map(x => `
      <tr>
        <td><strong>${esc(x.station_id)}</strong></td>
        <td>${esc(x.station_segment)}</td>
        <td>${Number(x.next_hour_demand_kwh).toFixed(2)} kWh</td>
        <td>${(Number(x.fault_probability) * 100).toFixed(2)}%</td>
        <td>${badge(x.fault_label)}</td>
      </tr>`).join("");
  } catch {
    body.innerHTML = `<tr><td colspan="5">Could not load station predictions. Check that the model artifacts are generated.</td></tr>`;
  }
}

async function loadMetrics() {
  try {
    const r = await fetch("/metrics");
    if (!r.ok) throw new Error("metrics failed");
    const j = await r.json();
    const rt = j.runtime || {};
    if ($("#requestCount")) $("#requestCount").textContent = Number(rt.prediction_requests || 0).toLocaleString();
    if ($("#driftCount")) $("#driftCount").textContent = Number(rt.drift_checks || 0).toLocaleString();
    if ($("#retrainCount")) $("#retrainCount").textContent = Number(rt.retrain_requests || 0).toLocaleString();
  } catch {
    ["#requestCount", "#driftCount", "#retrainCount"].forEach(id => { if ($(id)) $(id).textContent = "offline"; });
  }
}

function renderImportance(container, rows, valueKey, digits = 4) {
  if (!container) return;
  if (!rows?.length) {
    container.innerHTML = `<div class="loading">No importance artifact available.</div>`;
    return;
  }
  const max = Math.max(...rows.map(x => Math.max(0, Number(x[valueKey]) || 0)), 1e-12);
  container.innerHTML = rows.map(x => {
    const value = Number(x[valueKey]) || 0;
    const width = Math.max(0, Math.min(100, value / max * 100));
    return `<div class="importance-row">
      <span class="feature-name" title="${esc(x.feature)}">${esc(x.feature)}</span>
      <div class="importance-track"><div class="importance-fill" style="width:${width.toFixed(2)}%"></div></div>
      <span class="importance-value">${value.toFixed(digits)}</span>
    </div>`;
  }).join("");
}

let importanceLoaded = false;
async function loadFeatureImportance() {
  if (importanceLoaded) return;
  const reg = $("#regImportance");
  const clf = $("#clfImportance");
  if (!reg || !clf) return;
  try {
    const r = await fetch("/insights/feature-importance");
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "importance request failed");
    renderImportance(reg, j.regression_shap, "regression_mean_abs_shap", 4);
    renderImportance(clf, j.classification_rf_permutation, "pr_auc_drop_mean", 5);
    importanceLoaded = true;
  } catch (err) {
    reg.innerHTML = `<div class="loading">${esc(err.message)}</div>`;
    clf.innerHTML = `<div class="loading">${esc(err.message)}</div>`;
  }
}

const clusterColors = ["#48e7ae", "#55a7ff", "#ffd166", "#c58cff", "#ff8a80", "#78d6c6", "#ffae6d", "#8ce99a"];
function renderClusterPlot(data) {
  const box = $("#clusterPlot");
  const legend = $("#clusterLegend");
  if (!box || !legend) return;
  const points = data.points || [];
  if (!points.length) {
    box.innerHTML = `<div class="loading">No cluster points available.</div>`;
    return;
  }
  const width = 900, height = 420, pad = 44;
  const xs = points.map(p => Number(p.pca_1));
  const ys = points.map(p => Number(p.pca_2));
  let xmin = Math.min(...xs), xmax = Math.max(...xs), ymin = Math.min(...ys), ymax = Math.max(...ys);
  if (xmin === xmax) { xmin -= 1; xmax += 1; }
  if (ymin === ymax) { ymin -= 1; ymax += 1; }
  const sx = x => pad + (x - xmin) / (xmax - xmin) * (width - pad * 2);
  const sy = y => height - pad - (y - ymin) / (ymax - ymin) * (height - pad * 2);
  const grid = [0,.25,.5,.75,1].map(t => {
    const x = pad + t * (width - pad * 2);
    const y = pad + t * (height - pad * 2);
    return `<line class="scatter-grid" x1="${x}" y1="${pad}" x2="${x}" y2="${height-pad}"/><line class="scatter-grid" x1="${pad}" y1="${y}" x2="${width-pad}" y2="${y}"/>`;
  }).join("");
  const circles = points.map(p => {
    const seg = Number(p.kmeans_segment);
    const color = clusterColors[Math.abs(seg) % clusterColors.length];
    const title = `${p.station_id}\nK-Means segment ${seg}\nDBSCAN ${p.dbscan_segment}\nPCA (${Number(p.pca_1).toFixed(2)}, ${Number(p.pca_2).toFixed(2)})`;
    return `<circle class="scatter-point" cx="${sx(Number(p.pca_1)).toFixed(1)}" cy="${sy(Number(p.pca_2)).toFixed(1)}" r="6" fill="${color}"><title>${esc(title)}</title></circle>`;
  }).join("");
  box.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="PCA station cluster plot">${grid}<text class="scatter-axis" x="${width/2}" y="${height-10}" text-anchor="middle">PCA component 1</text><text class="scatter-axis" transform="translate(14 ${height/2}) rotate(-90)" text-anchor="middle">PCA component 2</text>${circles}</svg>`;
  const counts = data.kmeans_counts || {};
  legend.innerHTML = Object.keys(counts).sort((a,b) => Number(a)-Number(b)).map(k => {
    const color = clusterColors[Math.abs(Number(k)) % clusterColors.length];
    return `<span><i class="legend-dot" style="background:${color}"></i>Segment ${esc(k)} • ${esc(counts[k])} stations</span>`;
  }).join("");
}

let clustersLoaded = false;
async function loadClusters() {
  if (clustersLoaded) return;
  try {
    const r = await fetch("/insights/clusters");
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "cluster request failed");
    if ($("#clusterStations")) $("#clusterStations").textContent = j.stations;
    if ($("#clusterNoise")) $("#clusterNoise").textContent = j.noise_stations;
    renderClusterPlot(j);
    clustersLoaded = true;
  } catch (err) {
    if ($("#clusterPlot")) $("#clusterPlot").innerHTML = `<div class="loading">${esc(err.message)}</div>`;
  }
}

$("#refreshBtn")?.addEventListener("click", () => { loadStations(); loadMetrics(); });

$("#predictForm")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const num = (k) => Number(fd.get(k));
  const submit = e.target.querySelector("button[type='submit']");
  const oldText = submit.textContent;
  submit.disabled = true;
  submit.textContent = "Running models...";
  const chosenTimestamp = fd.get("timestamp");

  const payload = {
    station_id: fd.get("station_id"),
    timestamp: chosenTimestamp ? new Date(chosenTimestamp).toISOString() : new Date().toISOString(),
    latitude: num("latitude"),
    longitude: num("longitude"),
    session_count: Math.round(num("session_count")),
    avg_session_minutes: num("avg_session_minutes"),
    avg_total_duration_minutes: num("avg_total_duration_minutes"),
    avg_power_kw: num("avg_power_kw"),
    energy_kwh: num("energy_kwh"),
    fee_usd: num("fee_usd"),
    level1_share: num("level1_share"),
    level2_share: num("level2_share"),
    dc_fast_share: num("dc_fast_share"),
  };

  try {
    const r = await fetch("/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Prediction failed");

    $("#resultTitle").textContent = j.station_id;
    $("#demandOut").textContent = Number(j.next_hour_demand_kwh).toFixed(2);
    $("#faultProb").textContent = `${(Number(j.fault_probability) * 100).toFixed(2)}%`;
    $("#segmentOut").textContent = j.station_segment;
    $("#versionOut").textContent = j.model_version;
    $("#faultBar").style.width = `${Math.min(100, Number(j.fault_probability) * 100)}%`;
    $("#faultBar").style.background = isRisk(j.fault_label) ? "var(--danger)" : "var(--accent)";
    const status = $("#faultStatus");
    status.textContent = j.fault_label;
    status.className = `status ${isRisk(j.fault_label) ? "fault" : "ok"}`;
    loadMetrics();
  } catch (err) {
    $("#resultTitle").textContent = "Prediction error";
    $("#demandOut").textContent = "—";
    $("#faultProb").textContent = "—";
    const status = $("#faultStatus");
    status.textContent = err.message;
    status.className = "status fault";
  } finally {
    submit.disabled = false;
    submit.textContent = oldText;
  }
});

$("#runDriftBtn")?.addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const result = $("#driftResult");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Running PSI on real data...";
  result.className = "action-result neutral-box";
  result.textContent = "Reading latest station-hour observations and comparing them with the training baseline...";
  try {
    const r = await fetch("/drift/demo?limit=250", {method:"POST"});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "drift check failed");
    const ranked = Object.entries(j.feature_psi || {}).sort((a,b) => Number(b[1])-Number(a[1])).slice(0,5);
    const klass = j.status === "STABLE" ? "good" : (j.status === "MODERATE_DRIFT" ? "warn" : "bad");
    result.className = `action-result ${klass}`;
    result.innerHTML = `<strong>${esc(j.status)}</strong> • ${esc(j.observations)} real observations<br>Max PSI: <b>${Number(j.max_psi).toFixed(4)}</b> • Mean PSI: <b>${Number(j.mean_psi).toFixed(4)}</b><div class="drift-feature-list">${ranked.map(([k,v]) => `<span>${esc(k)} ${Number(v).toFixed(3)}</span>`).join("")}</div>${j.retrain_recommended ? '<br><b>Retraining recommended.</b>' : '<br>No automatic retraining recommendation.'}`;
    loadMetrics();
  } catch (err) {
    result.className = "action-result bad";
    result.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
});

$("#retrainBtn")?.addEventListener("click", async (e) => {
  if (!window.confirm("Record a retraining request for ChargeUp v2.0.0?")) return;
  const btn = e.currentTarget;
  const result = $("#retrainResult");
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Recording request...";
  try {
    const r = await fetch("/retrain/trigger", {method:"POST"});
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "retraining trigger failed");
    result.className = "action-result good";
    result.innerHTML = `<strong>${esc(j.status).toUpperCase()}</strong><br>${esc(j.message)}<br><code>${esc(j.flag_file || '')}</code>`;
    loadMetrics();
  } catch (err) {
    result.className = "action-result bad";
    result.textContent = err.message;
  } finally {
    btn.disabled = false;
    btn.textContent = old;
  }
});

setDefaultTimestamp();
health();
loadStations();
loadMetrics();
