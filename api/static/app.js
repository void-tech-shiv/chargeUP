const $ = (s) => document.querySelector(s);

async function health() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    $("#health").textContent = j.status === "ok" ? `● API ONLINE • ${j.version}` : "API ERROR";
    $("#health").className = "pill ok";
  } catch {
    $("#health").textContent = "● API OFFLINE";
    $("#health").className = "pill fault";
  }
}

function isRisk(label) {
  return label === "SERVER_TERMINATION_RISK" || label === "FAULT";
}

function badge(label) {
  return `<span class="badge ${isRisk(label) ? "fault" : "ok"}">${label}</span>`;
}

async function loadStations() {
  const body = $("#stationRows");
  body.innerHTML = `<tr><td colspan="5">Loading Palo Alto predictions…</td></tr>`;
  try {
    const r = await fetch("/demo/stations?limit=10");
    const data = await r.json();
    body.innerHTML = data.map(x => `
      <tr>
        <td><strong>${x.station_id}</strong></td>
        <td>${x.station_segment}</td>
        <td>${Number(x.next_hour_demand_kwh).toFixed(2)} kWh</td>
        <td>${(x.fault_probability * 100).toFixed(2)}%</td>
        <td>${badge(x.fault_label)}</td>
      </tr>
    `).join("");
  } catch (e) {
    body.innerHTML = `<tr><td colspan="5">Could not load station predictions.</td></tr>`;
  }
}

$("#refreshBtn").addEventListener("click", loadStations);

$("#predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const num = (k) => Number(fd.get(k));

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

  const r = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });

  if (!r.ok) {
    const err = await r.json();
    throw new Error(err.detail || "Prediction failed");
  }

  const j = await r.json();
  $("#resultTitle").textContent = j.station_id;
  $("#demandOut").textContent = Number(j.next_hour_demand_kwh).toFixed(2);
  $("#faultProb").textContent = `${(j.fault_probability * 100).toFixed(2)}%`;
  $("#faultBar").style.width = `${Math.min(100, j.fault_probability * 100)}%`;
  $("#faultBar").style.background = isRisk(j.fault_label) ? "var(--danger)" : "var(--accent)";

  const status = $("#faultStatus");
  status.textContent = j.fault_label;
  status.className = `status ${isRisk(j.fault_label) ? "fault" : "ok"}`;
});

health();
loadStations();
