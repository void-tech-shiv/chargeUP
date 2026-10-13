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
}

$$(".nav-item").forEach(btn => btn.addEventListener("click", () => showPage(btn.dataset.page)));
$("#menuBtn")?.addEventListener("click", () => $("#sidebar")?.classList.toggle("open"));

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

$("#refreshBtn")?.addEventListener("click", () => { loadStations(); loadMetrics(); });

$("#predictForm")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const num = (k) => Number(fd.get(k));
  const submit = e.target.querySelector("button[type='submit']");
  const oldText = submit.textContent;
  submit.disabled = true;
  submit.textContent = "Running models...";

  const payload = {
    station_id: fd.get("station_id"),
    timestamp: new Date().toISOString(),
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

health();
loadStations();
loadMetrics();
