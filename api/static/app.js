const $ = (s) => document.querySelector(s);

async function health() {
  try {
    const r = await fetch("/health");
    const j = await r.json();
    $("#health").textContent = j.status === "ok" ? "● API ONLINE" : "API ERROR";
    $("#health").className = "pill ok";
  } catch {
    $("#health").textContent = "● API OFFLINE";
    $("#health").className = "pill fault";
  }
}

function badge(label) {
  return `<span class="badge ${label === "FAULT" ? "fault" : "ok"}">${label}</span>`;
}

async function loadStations() {
  const body = $("#stationRows");
  body.innerHTML = `<tr><td colspan="5">Loading predictions…</td></tr>`;
  try {
    const r = await fetch("/demo/stations?limit=10");
    const data = await r.json();
    body.innerHTML = data.map(x => `
      <tr>
        <td><strong>${x.station_id}</strong></td>
        <td>${x.station_segment}</td>
        <td>${Number(x.next_hour_demand_kwh).toFixed(2)} kWh</td>
        <td>${(x.fault_probability * 100).toFixed(1)}%</td>
        <td>${badge(x.fault_label)}</td>
      </tr>
    `).join("");
  } catch (e) {
    body.innerHTML = `<tr><td colspan="5">Could not load demo stations.</td></tr>`;
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
    temperature_c: num("temperature_c"),
    humidity_pct: num("humidity_pct"),
    rainfall_mm: num("rainfall_mm"),
    latitude: 16.5062,
    longitude: 80.6480,
    session_count: Math.round(num("session_count")),
    avg_session_minutes: num("avg_session_minutes"),
    avg_power_kw: num("avg_power_kw"),
    connector_errors: Math.round(num("connector_errors")),
    voltage_v: num("voltage_v"),
    current_a: num("current_a"),
    availability_pct: num("availability_pct"),
    energy_kwh: num("energy_kwh"),
  };

  const r = await fetch("/predict", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const j = await r.json();

  $("#resultTitle").textContent = j.station_id;
  $("#demandOut").textContent = Number(j.next_hour_demand_kwh).toFixed(2);
  $("#faultProb").textContent = `${(j.fault_probability * 100).toFixed(1)}%`;
  $("#faultBar").style.width = `${Math.min(100, j.fault_probability * 100)}%`;
  $("#faultBar").style.background = j.fault_label === "FAULT" ? "var(--danger)" : "var(--accent)";

  const status = $("#faultStatus");
  status.textContent = j.fault_label;
  status.className = `status ${j.fault_label === "FAULT" ? "fault" : "ok"}`;
});

health();
loadStations();
