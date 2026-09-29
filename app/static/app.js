/* Beginner-friendly historical cyclone research dashboard. */
let currentStorm = null;
let availableStorms = [];
let globeView = null;

const statusLabels = {
  ready: "Ready", partial_coverage: "Partly available", waiting_for_raw_data: "Waiting for data",
  waiting_for_source_features: "Waiting for features", ready_for_feature_extraction: "Ready to extract",
  credentials_required: "Credentials needed", registration_required: "Registration needed", planned: "Planned",
  setup_required: "Setup required", ready_to_collect: "Ready to collect", collection_started: "Collection started",
  pending: "Not started", metadata_current: "Metadata current", stale: "Stale metadata",
  metadata_unverified: "Time unavailable", not_polled: "Not checked", unavailable: "Unavailable",
};

function number(value, digits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: digits }).format(value);
}
function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, char => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "'":"&#39;", '"':"&quot;" }[char]));
}
function shortTime(value) {
  return value ? value.replace("T", " ").replace("Z", " UTC") : "—";
}

function openView(id) {
  document.querySelectorAll(".view").forEach(view => {
    const active = view.id === id;
    view.hidden = !active;
    view.classList.toggle("active", active);
  });
  document.querySelectorAll("[data-view-target]").forEach(button => {
    button.classList.toggle("active", button.dataset.viewTarget === id);
  });
  if (id === "map-view" && globeView) setTimeout(() => globeView.resize(), 40);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setupNavigation() {
  document.querySelectorAll("[data-view-target]").forEach(button => {
    button.addEventListener("click", () => openView(button.dataset.viewTarget));
  });
}

function setupTheme() {
  const selector = document.querySelector("#theme-toggle");
  let savedTheme = "light";
  try { savedTheme = localStorage.getItem("cyclone-sentinel-theme") || "light"; } catch { /* Storage may be unavailable. */ }
  const applyTheme = theme => {
    const selectedTheme = theme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = selectedTheme;
    selector.checked = selectedTheme === "dark";
    if (globeView) globeView.applyTheme();
    try { localStorage.setItem("cyclone-sentinel-theme", selectedTheme); } catch { /* Keep the selected session theme. */ }
  };
  applyTheme(savedTheme);
  selector.addEventListener("change", () => applyTheme(selector.checked ? "dark" : "light"));
}

async function loadStormList(selectedId) {
  const response = await fetch("/api/v1/storms");
  if (!response.ok) throw new Error("The collected storm list is unavailable.");
  availableStorms = await response.json();
  const selector = document.querySelector("#storm-select");
  const selected = availableStorms.find(storm => storm.id === selectedId) || availableStorms[0];
  document.querySelector("#storm-select-label").textContent = selected ? `${selected.name} · ${selected.season}` : "No collected storms";
  selector.disabled = !selected;
  renderStormOptions(document.querySelector("#storm-search").value);
}

function renderStormOptions(query = "") {
  const normalized = query.trim().toLowerCase();
  const options = availableStorms.filter(storm => `${storm.name} ${storm.season} ${storm.peak_category}`.toLowerCase().includes(normalized));
  const container = document.querySelector("#storm-options");
  container.innerHTML = options.length ? options.map(storm => `
    <button class="storm-option" type="button" role="option" data-storm-id="${escapeHtml(storm.id)}" aria-selected="${storm.id === currentStorm?.id}">
      <span><strong>${escapeHtml(storm.name)}</strong><small>${escapeHtml(storm.peak_category)}</small></span>
      <em>${escapeHtml(storm.season)}</em>
    </button>`).join("") : "<p class=\"empty-storm-options\">No storm matches that search.</p>";
  container.querySelectorAll(".storm-option").forEach(option => option.onclick = () => {
    closeStormPicker();
    loadStorm(option.dataset.stormId);
  });
}

function closeStormPicker() {
  const menu = document.querySelector("#storm-select-menu");
  const button = document.querySelector("#storm-select");
  menu.hidden = true;
  button.setAttribute("aria-expanded", "false");
}

function setupStormPicker() {
  const button = document.querySelector("#storm-select");
  const menu = document.querySelector("#storm-select-menu");
  const search = document.querySelector("#storm-search");
  button.onclick = () => {
    const willOpen = menu.hidden;
    menu.hidden = !willOpen;
    button.setAttribute("aria-expanded", String(willOpen));
    if (willOpen) { search.focus(); renderStormOptions(search.value); }
  };
  search.oninput = () => renderStormOptions(search.value);
  search.onkeydown = event => { if (event.key === "Escape") { closeStormPicker(); button.focus(); } };
  document.addEventListener("click", event => {
    if (!event.target.closest(".storm-picker-control")) closeStormPicker();
  });
}

function renderOverview(storm) {
  const current = storm.current || {};
  const ri = storm.rapid_intensification || {};
  const dvorak = storm.dvorak || {};
  document.querySelector("#storm-name").textContent = storm.name;
  document.querySelector("#storm-subtitle").textContent = `${storm.status || "Storm"} · Last historical observation: ${shortTime(storm.last_updated)}`;
  document.querySelector("#replay-mode-text").textContent = storm.data_mode || "Historical replay";
  const rainfall = current.rainfall_mm_hr === null || current.rainfall_mm_hr === undefined ? "Not collected" : `${number(current.rainfall_mm_hr, 1)} mm/hr`;
  const cards = [
    ["Current wind", `${number(current.wind_kmph)} km/h`, "Observed best-track value"],
    ["Central pressure", `${number(current.pressure_hpa)} hPa`, "Observed best-track value"],
    ["Storm type", storm.status || "—", "Based on recorded wind"],
    ["Rainfall data", rainfall, "Blank means no matching IMERG feature is available"],
  ];
  document.querySelector("#metrics").innerHTML = cards.map(([label, value, detail]) => `<article class="metric"><span class="label">${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(detail)}</small></article>`).join("");
  renderAnalysis(dvorak, ri);
  renderCoastal(storm.landfall || {}, storm.district_risk || []);
}

function renderAnalysis(dvorak, ri) {
  const score = Math.round((ri.ri_score || 0) * 100);
  document.querySelector("#dvorak-t-num").textContent = `T${number(dvorak.t_number, 1)}`;
  document.querySelector("#dvorak-badge").textContent = dvorak.pattern_type || "Unavailable";
  document.querySelector("#dvorak-delta-p").textContent = `${number(dvorak.central_pressure_deficit_hpa, 1)} hPa`;
  document.querySelector("#dvorak-desc").textContent = dvorak.pattern_description || "No proxy detail is available.";
  document.querySelector("#ri-score").textContent = `${score}/100`;
  document.querySelector("#ri-alert-badge").textContent = ri.status_label || "Unavailable";
  document.querySelector("#ri-score-bar").style.width = `${score}%`;
  document.querySelector("#ri-summary").textContent = ri.summary || "No screening summary is available.";
  const factors = Object.values(ri.factor_scores || {});
  document.querySelector("#ri-factors").innerHTML = factors.map(factor => `<div class="factor"><span>${escapeHtml(factor.name)}</span><b>${escapeHtml(factor.value)}${factor.favorable ? " · favourable" : " · limiting"}</b></div>`).join("");
}

function renderCoastal(landfall, districts) {
  document.querySelector("#lf-landmark").textContent = landfall.nearest_landmark || "Open ocean";
  document.querySelector("#lf-port").textContent = landfall.nearest_port || "—";
  document.querySelector("#lf-eta-ist").textContent = landfall.eta_timestamp_ist || "No proximity flag";
  document.querySelector("#lf-surge").textContent = landfall.storm_surge_meters === null || landfall.storm_surge_meters === undefined ? "Not available" : `${number(landfall.storm_surge_meters, 1)} m proxy`;
  document.querySelector("#districts-list").innerHTML = districts.length ? districts.map(district => `<article class="district"><div><strong>${escapeHtml(district.district)}</strong><small>${number(district.distance_to_track_km)} km from research track · Static population: ${number(district.population_exposed)}</small></div><span class="tag">${escapeHtml(district.alert_tier)}</span></article>`).join("") : "<p>No static reference district is near this research route.</p>";
}

function globePoint(lat, lon, radius = 1) {
  const latitude = Number(lat) * Math.PI / 180;
  const longitude = Number(lon) * Math.PI / 180;
  return {
    x: radius * Math.cos(latitude) * Math.sin(longitude),
    y: radius * Math.sin(latitude),
    z: radius * Math.cos(latitude) * Math.cos(longitude),
  };
}

function globeArc(points, radius = 1) {
  if (points.length < 2) return points.map(point => globePoint(point.lat, point.lon, radius));
  const arc = [];
  points.forEach((point, index) => {
    if (index === points.length - 1) return;
    const next = points[index + 1];
    const start = globePoint(point.lat, point.lon);
    const end = globePoint(next.lat, next.lon);
    for (let step = 0; step < 10; step += 1) {
      const fraction = step / 10;
      const x = start.x + (end.x - start.x) * fraction;
      const y = start.y + (end.y - start.y) * fraction;
      const z = start.z + (end.z - start.z) * fraction;
      const length = Math.hypot(x, y, z) || 1;
      arc.push({ x: x / length * radius, y: y / length * radius, z: z / length * radius });
    }
  });
  const last = points.at(-1);
  arc.push(globePoint(last.lat, last.lon, radius));
  return arc;
}

function globeDistanceRing(lat, lon, distanceKm) {
  const earthRadiusKm = 6371;
  const angularDistance = Number(distanceKm) / earthRadiusKm;
  const latitude = Number(lat) * Math.PI / 180;
  const longitude = Number(lon) * Math.PI / 180;
  const points = [];
  for (let step = 0; step < 48; step += 1) {
    const bearing = (step / 48) * Math.PI * 2;
    const ringLat = Math.asin(Math.sin(latitude) * Math.cos(angularDistance) + Math.cos(latitude) * Math.sin(angularDistance) * Math.cos(bearing));
    const ringLon = longitude + Math.atan2(Math.sin(bearing) * Math.sin(angularDistance) * Math.cos(latitude), Math.cos(angularDistance) - Math.sin(latitude) * Math.sin(ringLat));
    points.push(globePoint(ringLat * 180 / Math.PI, ringLon * 180 / Math.PI));
  }
  return points;
}

function clearGlobeLayer(layer) { layer.items = []; }

function countryRings(geometry) {
  const polygons = geometry?.type === "Polygon" ? [geometry.coordinates] : geometry?.type === "MultiPolygon" ? geometry.coordinates : [];
  return polygons.flatMap(polygon => polygon.map(ring => {
    const simplified = ring.filter(([lon, lat], index) => index === 0 || index === ring.length - 1 || Math.abs(lon - ring[index - 1][0]) + Math.abs(lat - ring[index - 1][1]) > .45);
    return simplified.map(([lon, lat]) => globePoint(lat, lon));
  }));
}

async function loadCountryBoundaries(view) {
  try {
    const response = await fetch("/static/world-countries.geojson");
    if (!response.ok) throw new Error("Country-boundary data is unavailable.");
    const world = await response.json();
    view.countries = (world.features || []).flatMap(feature => countryRings(feature.geometry));
  } catch {
    view.countries = [];
  }
}

function initializeGlobe() {
  const host = document.querySelector("#globe");
  const canvas = document.createElement("canvas");
  canvas.setAttribute("aria-hidden", "true");
  host.replaceChildren(canvas);
  const context = canvas.getContext("2d");
  const view = {
    canvas, context, rotation: { x: 0, y: 0 }, homeRotation: { x: 0, y: 0 }, zoom: 1, zoomTarget: 1, countries: [],
    layers: { track: { visible: true, items: [] }, cone: { visible: true, items: [] }, radii: { visible: true, items: [] } },
  };
  let dragging = false;
  let lastPointer = null;
  let spinAgainAt = 0;
  let velocity = { x: 0, y: 0 };
  let pinchDistance = null;
  const touchPoints = new Map();
  const zoomInput = document.querySelector("#globe-zoom");
  const zoomValue = document.querySelector("#globe-zoom-value");
  const setZoom = value => {
    view.zoomTarget = Math.max(.65, Math.min(3.2, value));
    zoomInput.value = String(Math.round(view.zoomTarget * 100));
    zoomValue.value = `${Math.round(view.zoomTarget * 100)}%`;
    zoomValue.textContent = zoomValue.value;
  };
  const resetView = () => {
    velocity = { x: 0, y: 0 };
    view.rotation.x = view.homeRotation.x;
    view.rotation.y = view.homeRotation.y;
    setZoom(1);
  };
  document.querySelector("#globe-zoom-in").onclick = () => setZoom(view.zoomTarget + .2);
  document.querySelector("#globe-zoom-out").onclick = () => setZoom(view.zoomTarget - .2);
  document.querySelector("#globe-reset").onclick = resetView;
  zoomInput.oninput = () => setZoom(Number(zoomInput.value) / 100);
  const rotate = point => {
    const cosY = Math.cos(view.rotation.y); const sinY = Math.sin(view.rotation.y);
    const x = point.x * cosY - point.z * sinY; const z = point.x * sinY + point.z * cosY;
    const cosX = Math.cos(view.rotation.x); const sinX = Math.sin(view.rotation.x);
    return { x, y: point.y * cosX - z * sinX, z: point.y * sinX + z * cosX };
  };
  const project = point => {
    const rotated = rotate(point);
    const radius = Math.min(canvas.clientWidth, canvas.clientHeight) * .34 * view.zoom;
    return { x: canvas.clientWidth / 2 + rotated.x * radius, y: canvas.clientHeight / 2 - rotated.y * radius, visible: rotated.z >= 0 };
  };
  const path = (points, color, options = {}) => {
    const projected = points.map(project); let drawing = false;
    context.beginPath();
    projected.forEach(point => {
      if (!point.visible) { drawing = false; return; }
      if (drawing) context.lineTo(point.x, point.y); else context.moveTo(point.x, point.y);
      drawing = true;
    });
    context.strokeStyle = color; context.globalAlpha = options.opacity ?? 1; context.lineWidth = options.width ?? 1.5;
    context.setLineDash(options.dashed ? [6, 5] : []); context.stroke(); context.setLineDash([]); context.globalAlpha = 1;
  };
  const country = points => {
    const projected = points.map(project); const fullyVisible = projected.every(point => point.visible); let drawing = false;
    context.beginPath();
    projected.forEach(point => {
      if (!point.visible) { drawing = false; return; }
      if (drawing) context.lineTo(point.x, point.y); else context.moveTo(point.x, point.y);
      drawing = true;
    });
    if (fullyVisible) { context.closePath(); context.fill(); }
    context.stroke();
  };
  const render = () => {
    const width = canvas.clientWidth; const height = canvas.clientHeight; const radius = Math.min(width, height) * .34 * view.zoom;
    if (!width || !height) return;
    context.clearRect(0, 0, width, height);
    const dark = document.documentElement.dataset.theme === "dark";
    const glow = context.createRadialGradient(width * .38, height * .32, radius * .08, width / 2, height / 2, radius * 1.08);
    glow.addColorStop(0, dark ? "#287cb3" : "#58a4d2"); glow.addColorStop(.68, dark ? "#0c456f" : "#23658e"); glow.addColorStop(1, dark ? "#061c31" : "#153d5c");
    context.save(); context.shadowColor = dark ? "#3bb5f5" : "#73c7ef"; context.shadowBlur = Math.min(26, radius * .12); context.beginPath(); context.arc(width / 2, height / 2, radius, 0, Math.PI * 2); context.fillStyle = glow; context.fill(); context.restore();
    context.save(); context.beginPath(); context.arc(width / 2, height / 2, radius, 0, Math.PI * 2); context.clip();
    const terminator = context.createLinearGradient(width * .2, height * .18, width * .88, height * .8);
    terminator.addColorStop(0, "#ffffff00"); terminator.addColorStop(.54, "#00131a12"); terminator.addColorStop(1, dark ? "#0209169c" : "#03162780");
    context.fillStyle = terminator; context.fillRect(0, 0, width, height); context.restore();
    context.save(); context.beginPath(); context.arc(width / 2, height / 2, radius, 0, Math.PI * 2); context.clip();
    for (let latitude = -60; latitude <= 60; latitude += 30) path(globeArc(Array.from({ length: 73 }, (_, index) => ({ lat: latitude, lon: -180 + index * 5 }))), dark ? "#65a8d1" : "#a7d0e5", { opacity: .38, width: 1 });
    for (let longitude = -150; longitude < 180; longitude += 30) path(globeArc(Array.from({ length: 37 }, (_, index) => ({ lat: -90 + index * 5, lon: longitude }))), dark ? "#65a8d1" : "#a7d0e5", { opacity: .38, width: 1 });
    context.fillStyle = dark ? "#176a66" : "#3d8b71"; context.strokeStyle = dark ? "#9bd7c6" : "#c7ead9"; context.globalAlpha = .76; context.lineWidth = .75;
    view.countries.forEach(country);
    context.globalAlpha = 1;
    Object.values(view.layers).filter(layer => layer.visible).forEach(layer => layer.items.forEach(item => {
      if (item.type === "path") path(item.points, item.color, item);
      if (item.type === "marker") {
        const point = project(item.point); if (!point.visible) return;
        context.save(); context.shadowColor = item.ring ? "#72e4cf" : item.color; context.shadowBlur = item.ring ? 16 : 4;
        context.fillStyle = item.color; context.beginPath(); context.arc(point.x, point.y, item.radius, 0, Math.PI * 2); context.fill();
        if (item.ring) { context.strokeStyle = "#72e4cf"; context.lineWidth = 2.5; context.beginPath(); context.arc(point.x, point.y, item.radius + 11, 0, Math.PI * 2); context.stroke(); }
        if (item.label) { context.shadowBlur = 3; context.fillStyle = dark ? "#f1fbff" : "#08253c"; context.font = "700 11px Manrope, system-ui, sans-serif"; context.fillText(item.label, point.x + 16, point.y - 13); }
        context.restore();
      }
    }));
    context.restore();
    context.strokeStyle = dark ? "#75c9f7" : "#d2efff"; context.lineWidth = 1.5; context.beginPath(); context.arc(width / 2, height / 2, radius, 0, Math.PI * 2); context.stroke();
  };
  const resize = () => {
    const ratio = Math.min(window.devicePixelRatio || 1, 2); const width = Math.max(host.clientWidth, 1); const height = Math.max(host.clientHeight, 1);
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio); canvas.style.width = `${width}px`; canvas.style.height = `${height}px`; context.setTransform(ratio, 0, 0, ratio, 0, 0); render();
  };
  const distanceBetweenTouches = () => {
    const [first, second] = [...touchPoints.values()];
    return first && second ? Math.hypot(first.x - second.x, first.y - second.y) : null;
  };
  host.addEventListener("pointerdown", event => {
    touchPoints.set(event.pointerId, { x: event.clientX, y: event.clientY }); host.setPointerCapture(event.pointerId);
    if (touchPoints.size === 2) { dragging = false; velocity = { x: 0, y: 0 }; pinchDistance = distanceBetweenTouches(); return; }
    dragging = true; velocity = { x: 0, y: 0 }; lastPointer = { x: event.clientX, y: event.clientY };
  });
  host.addEventListener("pointermove", event => {
    if (touchPoints.has(event.pointerId)) touchPoints.set(event.pointerId, { x: event.clientX, y: event.clientY });
    if (touchPoints.size === 2) {
      const nextDistance = distanceBetweenTouches();
      if (nextDistance && pinchDistance) setZoom(view.zoomTarget + (nextDistance - pinchDistance) / 240);
      pinchDistance = nextDistance;
      return;
    }
    if (!dragging || !lastPointer) return;
    const changeX = event.clientX - lastPointer.x; const changeY = event.clientY - lastPointer.y;
    view.rotation.y -= changeX * .008; view.rotation.x = Math.max(-.75, Math.min(.75, view.rotation.x + changeY * .006));
    velocity = { x: -changeX * .00058, y: changeY * .00044 }; lastPointer = { x: event.clientX, y: event.clientY };
  });
  const finishDrag = event => {
    touchPoints.delete(event.pointerId);
    pinchDistance = touchPoints.size === 2 ? distanceBetweenTouches() : null;
    if (touchPoints.size === 1) { lastPointer = [...touchPoints.values()][0]; dragging = true; return; }
    dragging = false; lastPointer = null; spinAgainAt = performance.now() + 1400;
  };
  host.addEventListener("pointerup", finishDrag); host.addEventListener("pointercancel", finishDrag);
  host.addEventListener("wheel", event => { event.preventDefault(); setZoom(view.zoomTarget - event.deltaY * .0018); }, { passive: false });
  new ResizeObserver(resize).observe(host);
  const animate = now => {
    view.zoom += (view.zoomTarget - view.zoom) * .2;
    if (!dragging) {
      if (Math.abs(velocity.x) + Math.abs(velocity.y) > .00004) {
        view.rotation.y += velocity.x; view.rotation.x = Math.max(-.75, Math.min(.75, view.rotation.x + velocity.y)); velocity = { x: velocity.x * .93, y: velocity.y * .93 };
      } else if (now > spinAgainAt && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) view.rotation.y += .00065;
    }
    render(); requestAnimationFrame(animate);
  };
  resize(); void loadCountryBoundaries(view); requestAnimationFrame(animate);
  return { ...view, resize, setZoom, resetView, applyTheme: render };
}

function averageRadius(radii) {
  const values = Object.values(radii || {}).map(Number).filter(value => Number.isFinite(value) && value > 0);
  return values.length ? values.reduce((total, value) => total + value, 0) / values.length : null;
}

function addStormMarker(layer, lat, lon, label) {
  layer.items.push({ type: "marker", point: globePoint(lat, lon), radius: 8, color: "#f7fbff", ring: true, label });
}

function renderMap(storm) {
  if (!globeView) globeView = initializeGlobe();
  Object.values(globeView.layers).forEach(clearGlobeLayer);
  const current = storm.current || {};
  const observed = storm.observed_track || [];
  const lastObserved = observed.at(-1) || current;
  const forecast = [lastObserved, ...(storm.forecast_track || [])];
  if (observed.length > 1) globeView.layers.track.items.push({ type: "path", points: globeArc(observed), color: "#42d1a6", width: 4 });
  if (forecast.length > 1) globeView.layers.track.items.push({ type: "path", points: globeArc(forecast), color: "#5d9cff", dashed: true, width: 3 });
  const markerInterval = Math.max(1, Math.ceil(observed.length / 15));
  observed.forEach((point, index) => {
    if (index % markerInterval !== 0 && index !== observed.length - 1) return;
    globeView.layers.track.items.push({ type: "marker", point: globePoint(point.lat, point.lon), radius: index === observed.length - 1 ? 4 : 2, color: index === observed.length - 1 ? "#f7fbff" : "#42d1a6" });
  });
  if (Number.isFinite(Number(current.lat)) && Number.isFinite(Number(current.lon))) addStormMarker(globeView.layers.track, current.lat, current.lon, `${storm.name || "Storm"} · latest record`);
  if ((storm.cone_polygon || []).length > 2) {
    const polygon = storm.cone_polygon.map(([lat, lon]) => ({ lat, lon }));
    globeView.layers.cone.items.push({ type: "path", points: globeArc([...polygon, polygon[0]]), color: "#9bc6ff", dashed: true, opacity: .9, width: 1.6 });
  }
  const radii = current.wind_radii || {};
  [[radii.r34_km, "#f4b64f"], [radii.r50_km, "#ef8830"], [radii.r64_km, "#dd5d72"]].forEach(([quadrants, color]) => {
    const radius = averageRadius(quadrants);
    if (radius && Number.isFinite(Number(current.lat)) && Number.isFinite(Number(current.lon))) globeView.layers.radii.items.push({ type: "path", points: globeDistanceRing(current.lat, current.lon, radius), color, width: 1.6 });
  });
  if (Number.isFinite(Number(lastObserved.lon))) {
    globeView.homeRotation.x = -Number(lastObserved.lat || 0) * Math.PI / 180 * .22;
    globeView.homeRotation.y = Number(lastObserved.lon) * Math.PI / 180;
    globeView.rotation.x = globeView.homeRotation.x;
    globeView.rotation.y = globeView.homeRotation.y;
    globeView.setZoom(1);
  }
  document.querySelector("#forecast-list").innerHTML = (storm.forecast_track || []).map(point => `<article class="forecast-item"><strong>Research baseline · +${point.hours} hours</strong><small>${number(point.lat, 2)}°N, ${number(point.lon, 2)}°E · ${number(point.wind_kmph)} km/h · error radius ±${number(point.radius_km)} km</small></article>`).join("");
}

function setupMapButtons() {
  document.querySelectorAll(".map-toggle").forEach(button => button.onclick = () => {
    const layer = globeView?.layers[button.dataset.layer];
    if (!layer) return;
    layer.visible = !layer.visible;
    button.classList.toggle("active", layer.visible);
  });
}

function sourceCard(item, live = false) {
  const state = item.status || "pending";
  const label = statusLabels[state] || state.replaceAll("_", " ");
  const description = live ? `${item.provider || "Provider"} · ${item.mode || "Source product"}` : item.purpose || "Source status";
  const extra = live ? (item.latest?.published_at ? `Last published: ${shortTime(item.latest.published_at)}` : "No accepted current metadata") : "";
  return `<article class="source-card"><span class="status status-${escapeHtml(state)}">${escapeHtml(label)}</span><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(description)}</p><small>${escapeHtml(extra)}</small></article>`;
}

async function renderSources() {
  try {
    const [catalogResponse, liveResponse] = await Promise.all([fetch("/api/v1/data/catalog"), fetch("/api/v1/live/products")]);
    const catalog = await catalogResponse.json();
    const live = await liveResponse.json();
    document.querySelector("#pipeline-grid").innerHTML = (catalog.layers || []).map(item => sourceCard(item)).join("");
    document.querySelector("#pipeline-next").textContent = catalog.next_milestone || "No collection step is available.";
    document.querySelector("#live-grid").innerHTML = (live.products || []).map(item => sourceCard(item, true)).join("");
    document.querySelector("#live-next").textContent = live.next_step || "No provider status is available.";
  } catch {
    document.querySelector("#pipeline-next").textContent = "Data-source status could not be loaded.";
  }
}

function setupBrief() {
  const modal = document.querySelector("#bulletin-modal");
  const close = () => { modal.hidden = true; };
  document.querySelector("#btn-bulletin").onclick = () => { document.querySelector("#bulletin-content").textContent = currentStorm?.bulletin_text || "Research brief unavailable."; modal.hidden = false; };
  document.querySelector("#btn-close-modal").onclick = close;
  modal.onclick = event => { if (event.target === modal) close(); };
  document.querySelector("#btn-copy-bulletin").onclick = async () => { if (currentStorm?.bulletin_text) await navigator.clipboard.writeText(currentStorm.bulletin_text); };
  document.querySelector("#btn-download-bulletin").onclick = () => {
    const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([currentStorm?.bulletin_text || ""], { type:"text/plain" })); link.download = `cyclone-research-brief-${currentStorm?.id || "storm"}.txt`; link.click(); URL.revokeObjectURL(link.href);
  };
}

async function loadStorm(stormId) {
  const response = await fetch(`/api/v1/storms/${encodeURIComponent(stormId)}`);
  if (!response.ok) throw new Error("This historical replay could not be loaded.");
  currentStorm = await response.json();
  await loadStormList(currentStorm.id);
  renderOverview(currentStorm);
  renderMap(currentStorm);
  renderSources();
}

document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation(); setupTheme(); setupStormPicker(); setupBrief(); setupMapButtons();
  try { await loadStorm("current"); } catch (error) { document.querySelector("#storm-name").textContent = "Data unavailable"; document.querySelector("#storm-subtitle").textContent = error.message; }
});
