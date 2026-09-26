const stormId = "biparjoy-2023";

function number(value, maximumFractionDigits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits }).format(value);
}
function riskClass(level) { return level.toLowerCase(); }

function createMetrics(storm) {
  const values = [
    [storm.current.wind_kmph, "km/h", "Best-track wind", "Wind"],
    [storm.current.pressure_hpa, "hPa", "Best-track pressure", "Pressure"],
    [storm.model_metrics.track_error_km, "km", "24h track MAE", "Track"],
    [storm.model_metrics.intensity_mae_knots, "kt", "24h wind MAE", "Wind model"],
  ];
  document.querySelector("#metrics").innerHTML = values.map(([value, unit, label, eyebrow]) => `<article class="metric-card"><span class="metric-eyebrow">${eyebrow}</span><div class="metric-value">${number(value, 1)}${value === null || value === undefined ? "" : ` <small>${unit}</small>`}</div><div class="metric-label">${label}</div></article>`).join("");
}

function createDetails(storm) {
  document.querySelector("#storm-name").textContent = storm.name;
  document.querySelector("#storm-subtitle").textContent = `${storm.basin} · ${storm.status} · ${storm.data_mode}`;
  document.querySelector("#data-status").textContent = storm.data_mode.startsWith("Historical replay") ? "HISTORICAL REPLAY READY" : "DEMO DATASET";
  document.querySelector("#classification").textContent = storm.classification.label;
  const confidence = storm.classification.confidence;
  const confidenceTrack = document.querySelector(".confidence-track");
  confidenceTrack.hidden = confidence === null || confidence === undefined;
  document.querySelector("#confidence-bar").style.width = `${confidence * 100}%`;
  document.querySelector("#confidence-copy").textContent = storm.classification.detail || `${Math.round(confidence * 100)}% model confidence`;
  document.querySelector("#forecast-list").innerHTML = storm.forecast_track.map(point => `<div class="forecast-item"><span class="forecast-hour">+${point.hours}h</span><span>${point.lat.toFixed(1)}°N, ${point.lon.toFixed(1)}°E<br><span class="forecast-wind">±${point.radius_km} km baseline test uncertainty</span></span><b>${number(point.wind_kmph)}${point.wind_kmph === null ? "" : " km/h"}</b></div>`).join("");
  document.querySelector("#risk-grid").innerHTML = storm.district_risk.map(risk => `<article class="risk ${riskClass(risk.level)}"><h3>${risk.district}</h3>${risk.score === null ? "" : `<span class="risk-score">${risk.score}</span>`}<span class="risk-level">${risk.level}</span><ul>${risk.drivers.map(driver => `<li>${driver}</li>`).join("")}</ul></article>`).join("");
}

function createMap(storm) {
  const map = L.map("map", { zoomControl: false }).setView([19, 67], 5);
  L.control.zoom({ position: "bottomright" }).addTo(map);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 10, attribution: "© OpenStreetMap" }).addTo(map);
  const observed = storm.observed_track.map(p => [p.lat, p.lon]);
  const forecast = [observed.at(-1), ...storm.forecast_track.map(p => [p.lat, p.lon])];
  L.polyline(observed, { color: "#55e1c1", weight: 4 }).addTo(map);
  L.polyline(forecast, { color: "#ffbe4c", weight: 4, dashArray: "8 9" }).addTo(map);
  storm.observed_track.forEach((point, index) => L.circleMarker([point.lat, point.lon], { radius: index === storm.observed_track.length - 1 ? 8 : 4, color: "#55e1c1", fillColor: "#55e1c1", fillOpacity: 1 }).addTo(map));
  storm.forecast_track.forEach(point => {
    L.circle([point.lat, point.lon], { radius: point.radius_km * 1000, color: "#ffbe4c", weight: 1, fillColor: "#ffbe4c", fillOpacity: .08 }).addTo(map);
    L.circleMarker([point.lat, point.lon], { radius: 5, color: "#ffbe4c", fillColor: "#ffbe4c", fillOpacity: 1 }).bindTooltip(`+${point.hours}h · ${point.wind_kmph} km/h`).addTo(map);
  });
  map.fitBounds(L.latLngBounds([...observed, ...forecast]).pad(.35));
  new ResizeObserver(() => map.invalidateSize()).observe(document.querySelector("#map"));
}

async function init() {
  try {
    const [response, dataResponse] = await Promise.all([fetch(`/api/v1/storms/${stormId}`), fetch("/api/v1/data/status")]);
    if (!response.ok) throw new Error("Could not load storm data");
    const storm = await response.json();
    createMetrics(storm); createDetails(storm); createMap(storm);
    if (dataResponse.ok) {
      const dataset = await dataResponse.json();
      document.querySelector("#data-status").textContent = dataset.status === "ready" ? `DATASET READY · ${number(dataset.records)} OBSERVATIONS` : "DATASET PENDING";
    }
  } catch (error) { document.querySelector("#storm-name").textContent = "Dashboard unavailable"; document.querySelector("#storm-subtitle").textContent = error.message; }
}
init();
