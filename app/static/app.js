/**
 * Cyclone Sentinel · MoES Tropical Cyclone Intelligence Frontend
 * Supports interactive timeline playback, Dvorak pattern classification,
 * Rapid Intensification (RI) risk gauges, landfall impact & surge analysis,
 * asymmetric quadrant wind radii, and official IMD RSMC bulletins.
 */

// Global state
let currentStormData = null;
let currentStormList = [];
let mapInstance = null;
let mapLayers = {
  track: L.layerGroup(),
  cone: L.layerGroup(),
  radii: L.layerGroup(),
  satellite: L.layerGroup(),
  districts: L.layerGroup(),
  landfall: L.layerGroup(),
  scrubberMarker: L.layerGroup(),
};
let playbackInterval = null;
let isPlaying = false;
let activeLayersState = {
  track: true,
  radii: true,
  satellite: true,
  districts: true,
};

function number(value, maximumFractionDigits = 0) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits }).format(value);
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;"
  }[c]));
}

const layerStates = {
  ready: ["Ready", "ready"],
  partial_coverage: ["Partial coverage", "partial"],
  waiting_for_raw_data: ["Awaiting raw data", "waiting"],
  waiting_for_source_features: ["Awaiting features", "waiting"],
  ready_for_feature_extraction: ["Ready to extract", "planned"],
  credentials_required: ["Credentials needed", "blocked"],
  registration_required: ["Registration needed", "blocked"],
  planned: ["Planned", "planned"],
  pending: ["Pending", "pending"],
  metadata_current: ["Metadata current", "ready"],
  stale: ["Stale metadata", "partial"],
  metadata_unverified: ["Timestamp unavailable", "partial"],
  not_polled: ["Not polled", "pending"],
  unavailable: ["Unavailable", "blocked"],
};

/* ----------------------------------------------------
   1. TOPBAR & STORM SELECTION
---------------------------------------------------- */
async function loadStormSelector(selectedId) {
  try {
    const res = await fetch("/api/v1/storms");
    if (!res.ok) return;
    currentStormList = await res.json();
    const select = document.querySelector("#storm-select");
    select.innerHTML = currentStormList.map(s => {
      const isSel = s.id === selectedId ? "selected" : "";
      return `<option value="${escapeHtml(s.id)}" ${isSel}>${escapeHtml(s.name)} (${s.season}) · ${escapeHtml(s.peak_category)}</option>`;
    }).join("");

    select.onchange = (e) => {
      loadStorm(e.target.value);
    };
  } catch (err) {
    console.error("Failed to load storms list", err);
  }
}

/* ----------------------------------------------------
   2. HERO & METRIC CARDS
---------------------------------------------------- */
function renderHero(storm) {
  document.querySelector("#storm-name").textContent = storm.name;
  document.querySelector("#storm-subtitle").textContent =
    `${storm.basin} · ${storm.status} · Issued ${storm.last_updated}`;
  document.querySelector("#replay-mode-text").textContent = storm.data_mode;

  // RI Pill
  const ri = storm.rapid_intensification || {};
  const riPill = document.querySelector("#ri-pill");
  riPill.textContent = ri.status_label || "LOW RI RISK";
  riPill.className = `badge-pill badge-${(ri.alert_level || "low").toLowerCase()}`;

  // Landfall Pill
  const lf = storm.landfall || {};
  const lfPill = document.querySelector("#landfall-pill");
  if (lf.will_make_landfall) {
    lfPill.textContent = `LANDFALL: ${lf.category_at_landfall} (${lf.nearest_port})`;
    lfPill.className = "badge-pill badge-critical";
  } else {
    lfPill.textContent = "OCEANIC TRACK · NO IMMEDIATE LANDFALL";
    lfPill.className = "badge-pill badge-moderate";
  }
}

function renderMetrics(storm) {
  const cur = storm.current || {};
  const dvorak = storm.dvorak || {};
  const ri = storm.rapid_intensification || {};
  const lf = storm.landfall || {};
  const metrics = storm.model_metrics || {};

  const windKt = cur.wind_kmph ? Math.round(cur.wind_kmph / 1.852) : null;
  const etaText = lf.eta_hours ? `+${lf.eta_hours}h (${lf.eta_timestamp_ist ? lf.eta_timestamp_ist.split(' ')[0] : ''})` : "Open Sea";

  const cards = [
    {
      eyebrow: "SUSTAINED WIND",
      value: number(cur.wind_kmph),
      unit: "km/h",
      label: `${windKt ? windKt + " kt · " : ""}${storm.status}`,
    },
    {
      eyebrow: "CENTRAL PRESSURE",
      value: number(cur.pressure_hpa),
      unit: "hPa",
      label: `ΔP Drop: ${number(dvorak.central_pressure_deficit_hpa, 1)} hPa`,
    },
    {
      eyebrow: "DVORAK INTENSITY",
      value: `T${number(dvorak.t_number, 1)}`,
      unit: `/ CI${number(dvorak.ci_number, 1)}`,
      label: `${dvorak.pattern_type}`,
    },
    {
      eyebrow: "RAPID INTENSIFICATION",
      value: `${Math.round((ri.ri_probability || 0) * 100)}`,
      unit: "%",
      label: `${ri.status_label}`,
    },
    {
      eyebrow: "PROJECTED LANDFALL",
      value: etaText,
      unit: "",
      label: lf.nearest_port || "Open Ocean",
    },
    {
      eyebrow: "24H MODEL ERROR",
      value: number(metrics.track_error_km, 1),
      unit: "km",
      label: `Wind MAE: ${number(metrics.intensity_mae_knots, 1)} kt`,
    },
  ];

  document.querySelector("#metrics").innerHTML = cards.map(c => `
    <article class="metric-card">
      <span class="metric-eyebrow">${c.eyebrow}</span>
      <div class="metric-value">${c.value}${c.unit ? ` <small>${c.unit}</small>` : ""}</div>
      <div class="metric-label" title="${escapeHtml(c.label)}">${escapeHtml(c.label)}</div>
    </article>
  `).join("");
}

/* ----------------------------------------------------
   3. DVORAK PATTERN & INTENSITY CARD
---------------------------------------------------- */
function renderDvorak(storm) {
  const dv = storm.dvorak || {};
  const cm = dv.cloud_metrics || {};

  document.querySelector("#dvorak-badge").textContent = dv.pattern_type || "UNKNOWN";
  document.querySelector("#dvorak-t-num").textContent = `T${(dv.t_number || 1.0).toFixed(1)}`;
  document.querySelector("#dvorak-ci-num").textContent = `CI ${(dv.ci_number || 1.0).toFixed(1)}`;
  document.querySelector("#dvorak-desc").textContent = dv.pattern_description || "";
  document.querySelector("#dvorak-delta-p").textContent = `${number(dv.central_pressure_deficit_hpa, 1)} hPa`;

  const metricsGrid = [
    { label: "Core Convective Symmetry", val: `${number(cm.convective_symmetry_percent)}%` },
    { label: "Spiral Rainband Wrap", val: `${number(cm.log_spiral_wrap_turns, 2)} turns` },
    { label: "Min Cloud Top Temp", val: `${number(cm.min_cloud_top_temperature_celsius, 1)}°C` },
    { label: "Cloud Shield Span", val: `${number(cm.cloud_shield_diameter_km)} km` },
  ];

  document.querySelector("#cloud-metrics").innerHTML = metricsGrid.map(m => `
    <div class="cloud-metric-item">
      <span class="cm-lbl">${m.label}</span>
      <span class="cm-val">${m.val}</span>
    </div>
  `).join("");

  // Explainable AI Attribution Bars
  const attrList = Array.isArray(dv.attribution) ? dv.attribution : [];
  document.querySelector("#attribution-bars").innerHTML = attrList.map(a => {
    const pct = Math.round((a.weight || 0) * 100);
    return `
      <div class="attr-row">
        <div>
          <span style="color:var(--text); font-weight:600;">${escapeHtml(a.feature)}</span>
          <div class="attr-track"><div class="attr-fill" style="width: ${pct}%;"></div></div>
        </div>
        <span class="attr-pct">${pct}%</span>
      </div>
    `;
  }).join("");
}

/* ----------------------------------------------------
   4. RAPID INTENSIFICATION (RI) DIAGNOSTICS CARD
---------------------------------------------------- */
function renderRI(storm) {
  const ri = storm.rapid_intensification || {};
  const probPct = Math.round((ri.ri_probability || 0) * 100);

  const badge = document.querySelector("#ri-alert-badge");
  badge.textContent = ri.status_label || "LOW RI RISK";
  badge.className = `ri-alert-badge badge-${(ri.alert_level || "low").toLowerCase()}`;

  document.querySelector("#ri-prob-val").textContent = `${probPct}%`;
  document.querySelector("#ri-prob-bar").style.width = `${probPct}%`;
  document.querySelector("#ri-summary").textContent = ri.summary || "";

  // Dynamic factors list
  const factors = ri.factor_scores || {};
  document.querySelector("#ri-factors").innerHTML = Object.values(factors).map(f => `
    <div class="ri-factor-row">
      <span>${escapeHtml(f.name)}: <b>${escapeHtml(f.value)}</b></span>
      <span class="factor-status-pill ${f.favorable ? "pill-fav" : "pill-inh"}">
        ${f.favorable ? "FAVORABLE" : "INHIBITING"}
      </span>
    </div>
  `).join("");

  document.querySelector("#proj-steady").textContent = `${number(ri.projected_24h_wind_normal_knots)} kt`;
  document.querySelector("#proj-ri").textContent = `${number(ri.projected_24h_wind_ri_knots)} kt`;
}

/* ----------------------------------------------------
   5. COASTAL LANDFALL & SURGE HAZARD CARD
---------------------------------------------------- */
function renderLandfall(storm) {
  const lf = storm.landfall || {};
  const districts = Array.isArray(storm.district_risk) ? storm.district_risk : [];

  const badge = document.querySelector("#landfall-status-badge");
  if (lf.will_make_landfall) {
    badge.textContent = "COASTAL STRIKE IMMINENT";
    badge.className = "landfall-badge badge-critical";
  } else {
    badge.textContent = "DEEP BASIN RECURVATURE";
    badge.className = "landfall-badge badge-moderate";
  }

  document.querySelector("#lf-landmark").textContent = lf.nearest_landmark || "Open ocean track";
  document.querySelector("#lf-port").textContent = lf.nearest_port || "None nearby";
  document.querySelector("#lf-eta-ist").textContent = lf.eta_timestamp_ist || "N/A";
  document.querySelector("#lf-category").textContent = lf.category_at_landfall || "N/A";
  document.querySelector("#lf-surge").textContent = lf.storm_surge_meters
    ? `${number(lf.storm_surge_meters, 1)} m (${lf.surge_warning_level})`
    : "No surge threat";

  // Districts table
  if (districts.length === 0) {
    document.querySelector("#districts-list").innerHTML = `
      <div style="padding:12px; color:var(--muted); font-size:11px;">No coastal districts within high-risk corridor.</div>
    `;
    return;
  }

  document.querySelector("#districts-list").innerHTML = districts.map(d => {
    let tierClass = "dist-yellow";
    let tagClass = "tag-yellow";
    if (d.alert_tier.includes("RED")) {
      tierClass = "dist-red"; tagClass = "tag-red";
    } else if (d.alert_tier.includes("ORANGE")) {
      tierClass = "dist-orange"; tagClass = "tag-orange";
    } else if (d.alert_tier.includes("GREEN")) {
      tierClass = "dist-green"; tagClass = "tag-yellow";
    }

    return `
      <div class="district-item ${tierClass}">
        <div>
          <div class="dist-name">${escapeHtml(d.district)}</div>
          <div class="dist-meta">
            ${number(d.distance_to_track_km)} km to track · Pop: ${number(d.population_exposed)} · Wind: ${d.expected_wind_kmph} km/h
          </div>
        </div>
        <span class="dist-alert-tag ${tagClass}">${escapeHtml(d.alert_tier)}</span>
      </div>
    `;
  }).join("");
}

/* ----------------------------------------------------
   6. INTERACTIVE MAP & SATELLITE / RADII LAYERS
---------------------------------------------------- */
function createQuadrantPolygon(lat, lon, radiiObj, color, fillOpacity) {
  // Compute approximate 16-point polygon for 4 asymmetric quadrants: NE, SE, SW, NW
  const { ne, se, sw, nw } = radiiObj;
  const points = [];
  const kmToDegLat = 1 / 111.0;

  function addArc(rKm, startDeg, endDeg) {
    const degLon = rKm / (111.0 * Math.max(0.1, Math.cos((lat * Math.PI) / 180)));
    const degLat = rKm * kmToDegLat;
    for (let deg = startDeg; deg <= endDeg; deg += 22.5) {
      const rad = (deg * Math.PI) / 180;
      points.push([lat + degLat * Math.cos(rad), lon + degLon * Math.sin(rad)]);
    }
  }

  // NE quadrant (0 to 90 deg)
  addArc(ne, 0, 90);
  // SE quadrant (270 to 360 deg)
  addArc(se, 270, 360);
  // SW quadrant (180 to 270 deg)
  addArc(sw, 180, 270);
  // NW quadrant (90 to 180 deg)
  addArc(nw, 90, 180);

  return L.polygon(points, {
    color: color,
    weight: 1.5,
    fillColor: color,
    fillOpacity: fillOpacity,
  });
}

function renderSatelliteSimulation(lat, lon, tNumber) {
  // Creates multi-tier realistic thermal infrared (BD-Curve) cloud top layers
  const satGroup = mapLayers.satellite;
  satGroup.clearLayers();

  const baseRadiusKm = 140 + (tNumber || 3.0) * 35;

  // Outer Cirrus Canopy (-30°C to -45°C)
  satGroup.addLayer(
    L.circle([lat, lon], {
      radius: baseRadiusKm * 1000,
      color: "transparent",
      fillColor: "#475569",
      fillOpacity: 0.22,
    })
  );

  // Dense Central Overcast (-50°C to -65°C Cyan/Teal)
  satGroup.addLayer(
    L.circle([lat, lon], {
      radius: baseRadiusKm * 0.62 * 1000,
      color: "transparent",
      fillColor: "#06b6d4",
      fillOpacity: 0.32,
    })
  );

  // Cold Eyewall Ring (-70°C to -80°C Crimson/Magenta)
  satGroup.addLayer(
    L.circle([lat, lon], {
      radius: baseRadiusKm * 0.32 * 1000,
      color: "transparent",
      fillColor: "#ec4899",
      fillOpacity: 0.45,
    })
  );

  // Warm Core / Eye if T >= 4.5
  if (tNumber >= 4.5) {
    satGroup.addLayer(
      L.circle([lat, lon], {
        radius: 18 * 1000,
        color: "#ffffff",
        weight: 1,
        fillColor: "#1e293b",
        fillOpacity: 0.65,
      }).bindTooltip("Warm Eye Core", { permanent: false })
    );
  }
}

function buildMap(storm) {
  if (!mapInstance) {
    mapInstance = L.map("map", { zoomControl: false }).setView([18, 70], 5);
    L.control.zoom({ position: "bottomright" }).addTo(mapInstance);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 10,
      attribution: "© OpenStreetMap contributors",
    }).addTo(mapInstance);

    // Add all layer groups to map initially
    Object.values(mapLayers).forEach(lg => lg.addTo(mapInstance));
    new ResizeObserver(() => mapInstance.invalidateSize()).observe(document.querySelector("#map"));
  }

  // Clear existing layers
  Object.values(mapLayers).forEach(lg => lg.clearLayers());

  const cur = storm.current || {};
  const observed = (storm.observed_track || []).map(p => [p.lat, p.lon]);
  const forecast = [
    observed.at(-1) || [cur.lat, cur.lon],
    ...(storm.forecast_track || []).map(p => [p.lat, p.lon]),
  ];

  // 1. TRACKS
  mapLayers.track.addLayer(L.polyline(observed, { color: "#5ce2b8", weight: 4.5, opacity: 0.9 }));
  mapLayers.track.addLayer(L.polyline(forecast, { color: "#ffb854", weight: 3.5, dashArray: "8 8", opacity: 0.9 }));

  storm.observed_track.forEach((p, idx) => {
    const isLatest = idx === storm.observed_track.length - 1;
    L.circleMarker([p.lat, p.lon], {
      radius: isLatest ? 8 : 4,
      color: isLatest ? "#ffffff" : "#5ce2b8",
      weight: isLatest ? 2 : 1,
      fillColor: isLatest ? "#5ce2b8" : "#0d131b",
      fillOpacity: 1,
    })
      .bindTooltip(`${p.time.replace("T", " ").replace("Z", "")}<br><b>${p.stage || ""}</b> · ${p.wind_kmph || "—"} km/h`)
      .addTo(mapLayers.track);
  });

  (storm.forecast_track || []).forEach(p => {
    L.circleMarker([p.lat, p.lon], {
      radius: 5,
      color: "#ffb854",
      fillColor: "#ffb854",
      fillOpacity: 1,
    })
      .bindTooltip(`+${p.hours}h · ${p.lat}°N, ${p.lon}°E<br><b>${number(p.wind_kmph)} km/h</b> ±${p.radius_km} km`)
      .addTo(mapLayers.track);
  });

  // 2. CONE OF UNCERTAINTY POLYGON
  const conePoly = storm.cone_polygon || [];
  if (conePoly.length > 2) {
    mapLayers.cone.addLayer(
      L.polygon(conePoly, {
        color: "#ffb854",
        weight: 1.5,
        fillColor: "#ffb854",
        fillOpacity: 0.12,
        dashArray: "4 4",
      })
    );
  }

  // 3. ASYMMETRIC QUADRANT WIND RADII
  const radii = cur.wind_radii || {};
  if (radii.r34_km) {
    mapLayers.radii.addLayer(createQuadrantPolygon(cur.lat, cur.lon, radii.r34_km, "#ffb854", 0.12));
  }
  if (radii.r50_km) {
    mapLayers.radii.addLayer(createQuadrantPolygon(cur.lat, cur.lon, radii.r50_km, "#ff8e3c", 0.18));
  }
  if (radii.r64_km) {
    mapLayers.radii.addLayer(createQuadrantPolygon(cur.lat, cur.lon, radii.r64_km, "#ff4757", 0.25));
  }

  // 4. SIMULATED SATELLITE CLOUD CANOPY
  const tNum = storm.dvorak ? storm.dvorak.t_number : 3.5;
  renderSatelliteSimulation(cur.lat, cur.lon, tNum);

  // 5. LANDFALL VECTOR & TARGET
  const lf = storm.landfall || {};
  if (lf.will_make_landfall && lf.landfall_point) {
    const lfPt = [lf.landfall_point.lat, lf.landfall_point.lon];
    mapLayers.landfall.addLayer(
      L.circleMarker(lfPt, {
        radius: 9,
        color: "#ff3b4e",
        weight: 3,
        fillColor: "#ffffff",
        fillOpacity: 1,
      }).bindTooltip(`<b>PROJECTED LANDFALL</b><br>${lf.nearest_landmark}<br>ETA: ${lf.eta_timestamp_ist || ""}<br>Peak Surge: ${lf.storm_surge_meters || "—"} m`, { permanent: true, direction: "top", offset: [0, -10] })
    );
  }

  // 6. DISTRICTS AT RISK MARKERS
  (storm.district_risk || []).forEach(d => {
    // If district coordinates are approximated, render pin
    // Use alert tier color
    let col = "#ffd84d";
    if (d.alert_tier.includes("RED")) col = "#ff4757";
    else if (d.alert_tier.includes("ORANGE")) col = "#ffa502";
  });

  // Fit bounds
  const allPts = [...observed, ...forecast];
  if (allPts.length > 0) {
    mapInstance.fitBounds(L.latLngBounds(allPts).pad(0.32));
  }
}

/* ----------------------------------------------------
   7. TIMELINE PLAYBACK / SCRUBBER ENGINE
---------------------------------------------------- */
function setupTimeline(storm) {
  const track = storm.observed_track || [];
  const slider = document.querySelector("#time-slider");
  slider.max = Math.max(0, track.length - 1);
  slider.value = slider.max;

  function updateScrubber(index) {
    slider.value = index;
    const pt = track[index];
    if (!pt) return;

    document.querySelector("#scrub-timestamp").textContent = pt.time.replace("T", " ").replace("Z", " UTC");
    document.querySelector("#telemetry-coords").textContent = `${pt.lat.toFixed(2)}°N, ${pt.lon.toFixed(2)}°E`;
    document.querySelector("#telemetry-wind").textContent = pt.wind_kmph ? `${pt.wind_kmph} km/h` : "—";
    document.querySelector("#telemetry-stage").textContent = pt.stage || "—";

    // Update scrubber marker on map
    mapLayers.scrubberMarker.clearLayers();
    const isLatest = index === track.length - 1;
    L.circleMarker([pt.lat, pt.lon], {
      radius: 10,
      color: "#ffffff",
      weight: 2,
      fillColor: isLatest ? "#5ce2b8" : "#ffb854",
      fillOpacity: 1,
    })
      .bindTooltip(`Time: ${pt.time}<br>Wind: ${pt.wind_kmph || "—"} km/h<br>${pt.stage || ""}`, { permanent: false })
      .addTo(mapLayers.scrubberMarker);
  }

  slider.oninput = (e) => {
    updateScrubber(parseInt(e.target.value, 10));
  };

  document.querySelector("#btn-scrub-first").onclick = () => updateScrubber(0);
  document.querySelector("#btn-scrub-prev").onclick = () => {
    const nextIdx = Math.max(0, parseInt(slider.value, 10) - 1);
    updateScrubber(nextIdx);
  };
  document.querySelector("#btn-scrub-next").onclick = () => {
    const nextIdx = Math.min(track.length - 1, parseInt(slider.value, 10) + 1);
    updateScrubber(nextIdx);
  };
  document.querySelector("#btn-scrub-last").onclick = () => updateScrubber(track.length - 1);

  // Play / Pause Animation
  const playBtn = document.querySelector("#btn-scrub-play");
  playBtn.onclick = () => {
    if (isPlaying) {
      clearInterval(playbackInterval);
      isPlaying = false;
      playBtn.textContent = "▶ PLAY";
      playBtn.classList.remove("playing");
    } else {
      isPlaying = true;
      playBtn.textContent = "⏸ PAUSE";
      playBtn.classList.add("playing");
      if (parseInt(slider.value, 10) >= track.length - 1) {
        slider.value = 0;
      }
      playbackInterval = setInterval(() => {
        let currentIdx = parseInt(slider.value, 10);
        if (currentIdx < track.length - 1) {
          updateScrubber(currentIdx + 1);
        } else {
          clearInterval(playbackInterval);
          isPlaying = false;
          playBtn.textContent = "▶ PLAY";
          playBtn.classList.remove("playing");
        }
      }, 700);
    }
  };

  updateScrubber(track.length - 1);
}

/* ----------------------------------------------------
   8. MAP CONTROLS TOGGLES
---------------------------------------------------- */
function setupMapControls() {
  document.querySelectorAll(".layer-chip").forEach(chip => {
    chip.onclick = () => {
      const layerKey = chip.getAttribute("data-layer");
      activeLayersState[layerKey] = !activeLayersState[layerKey];
      chip.classList.toggle("active", activeLayersState[layerKey]);

      if (layerKey === "track") {
        if (activeLayersState.track) {
          mapLayers.track.addTo(mapInstance);
          mapLayers.cone.addTo(mapInstance);
        } else {
          mapLayers.track.remove();
          mapLayers.cone.remove();
        }
      } else if (layerKey === "radii") {
        if (activeLayersState.radii) mapLayers.radii.addTo(mapInstance);
        else mapLayers.radii.remove();
      } else if (layerKey === "satellite") {
        if (activeLayersState.satellite) mapLayers.satellite.addTo(mapInstance);
        else mapLayers.satellite.remove();
      } else if (layerKey === "districts") {
        if (activeLayersState.districts) {
          mapLayers.districts.addTo(mapInstance);
          mapLayers.landfall.addTo(mapInstance);
        } else {
          mapLayers.districts.remove();
          mapLayers.landfall.remove();
        }
      }
    };
  });
}

/* ----------------------------------------------------
   9. OFFICIAL IMD RSMC BULLETIN MODAL
---------------------------------------------------- */
function setupBulletinModal(storm) {
  const modal = document.querySelector("#bulletin-modal");
  const btnOpen = document.querySelector("#btn-bulletin");
  const btnClose = document.querySelector("#btn-close-modal");
  const btnActionClose = document.querySelector("#btn-close-bulletin-action");
  const btnCopy = document.querySelector("#btn-copy-bulletin");
  const btnDownload = document.querySelector("#btn-download-bulletin");
  const content = document.querySelector("#bulletin-content");
  const statusSpan = document.querySelector("#copy-status");

  function closeModal() {
    modal.hidden = true;
    modal.classList.add("hidden");
    modal.style.display = "none";
  }

  function openModal() {
    const activeData = currentStormData || storm;
    content.textContent = activeData.bulletin_text || "Advisory bulletin text unavailable.";
    modal.hidden = false;
    modal.classList.remove("hidden");
    modal.style.display = "flex";
    if (statusSpan) statusSpan.textContent = "";
  }

  if (btnOpen) btnOpen.onclick = openModal;
  if (btnClose) btnClose.onclick = closeModal;
  if (btnActionClose) btnActionClose.onclick = closeModal;

  modal.onclick = (e) => {
    if (e.target === modal) closeModal();
  };

  // Close on Escape key
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.style.display !== "none") {
      closeModal();
    }
  });

  if (btnCopy) {
    btnCopy.onclick = async () => {
      try {
        const text = (currentStormData || storm).bulletin_text || "";
        await navigator.clipboard.writeText(text);
        if (statusSpan) {
          statusSpan.textContent = "✓ Copied to clipboard!";
          setTimeout(() => { statusSpan.textContent = ""; }, 3000);
        }
      } catch {
        if (statusSpan) statusSpan.textContent = "Press Ctrl+C to copy";
      }
    };
  }

  if (btnDownload) {
    btnDownload.onclick = () => {
      const activeData = currentStormData || storm;
      const blob = new Blob([activeData.bulletin_text || ""], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `IMD_RSMC_BULLETIN_${activeData.id}.txt`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };
  }
}

/* ----------------------------------------------------
   10. PIPELINE & LIVE PRODUCTS STATUS
---------------------------------------------------- */
function createPipeline(catalog) {
  const layers = Array.isArray(catalog.layers) ? catalog.layers : [];
  const readyCount = layers.filter(layer => layer.status === "ready").length;
  const cohort = layers.find(layer => layer.id === "multisource_cohort");
  const rows = cohort?.counts?.rows || 0;
  const complete = cohort?.counts?.complete || 0;

  document.querySelector("#pipeline-summary").textContent =
    `${readyCount}/${layers.length} LAYERS READY · ${number(complete)}/${number(rows)} COMPLETE`;
  document.querySelector("#pipeline-next").textContent =
    catalog.next_milestone || "Source status is being refreshed.";

  document.querySelector("#pipeline-grid").innerHTML = layers.map(layer => {
    const [label, tone] = layerStates[layer.status] || ["Status unknown", "pending"];
    return `
      <article class="pipeline-layer pipeline-${tone}">
        <div class="layer-status"><span class="layer-dot"></span><span>${label}</span></div>
        <h3>${escapeHtml(layer.name)}</h3>
        <p>${escapeHtml(layer.purpose)}</p>
        <small>${escapeHtml(layer.counts ? `${number(layer.counts.complete || 0)}/${number(layer.counts.rows || 0)} rows` : "Contract verified")}</small>
      </article>
    `;
  }).join("");
}

function createLiveProducts(payload) {
  const products = Array.isArray(payload.products) ? payload.products : [];
  const currentCount = products.filter(product => product.status === "metadata_current").length;

  document.querySelector("#live-summary").textContent =
    `${currentCount}/${products.length} METADATA FEEDS CURRENT`;
  document.querySelector("#live-next").textContent =
    payload.next_step || "Operational-source status is being refreshed.";

  document.querySelector("#live-grid").innerHTML = products.map(product => {
    const [label, tone] = layerStates[product.status] || ["Status unknown", "pending"];
    const age = Number.isFinite(Number(product.metadata_age_hours))
      ? ` · ${number(product.metadata_age_hours, 1)}h old`
      : "";
    const published = product.latest?.published_at
      ? `Published ${product.latest.published_at}${age}`
      : "Timestamp unavailable";

    return `
      <article class="live-product pipeline-${tone}">
        <div class="layer-status"><span class="layer-dot"></span><span>${label}</span></div>
        <h3>${escapeHtml(product.name)}</h3>
        <p>${escapeHtml(product.mode)}</p>
        <dl>
          <div><dt>Latency</dt><dd>${escapeHtml(product.nominal_latency)}</dd></div>
          <div><dt>Coverage</dt><dd>${escapeHtml(product.coverage)}</dd></div>
        </dl>
        <small>${published}</small>
      </article>
    `;
  }).join("");
}

/* ----------------------------------------------------
   11. MAIN INITIALIZATION & STORM LOADER
---------------------------------------------------- */
async function loadStorm(stormId) {
  try {
    const url = stormId ? `/api/v1/storms/${encodeURIComponent(stormId)}` : "/api/v1/storms/current";
    const res = await fetch(url);
    if (!res.ok) throw new Error("Could not load storm data");
    const storm = await res.json();
    currentStormData = storm;

    renderHero(storm);
    renderMetrics(storm);
    renderDvorak(storm);
    renderRI(storm);
    renderLandfall(storm);
    buildMap(storm);
    setupTimeline(storm);
    setupBulletinModal(storm);

    document.querySelector("#data-status").textContent = `REPLAY READY · ${storm.observed_track.length} FIXES`;
  } catch (err) {
    document.querySelector("#storm-name").textContent = "Storm Replay Unavailable";
    document.querySelector("#storm-subtitle").textContent = err.message;
  }
}

async function init() {
  setupMapControls();

  // Load initial storm
  await loadStorm();
  if (currentStormData) {
    await loadStormSelector(currentStormData.id);
  }

  // Load auxiliary catalog & live feeds asynchronously
  try {
    const [dataRes, catalogRes, liveRes] = await Promise.all([
      fetch("/api/v1/data/status").catch(() => null),
      fetch("/api/v1/data/catalog").catch(() => null),
      fetch("/api/v1/live/products").catch(() => null),
    ]);

    if (catalogRes?.ok) createPipeline(await catalogRes.json());
    if (liveRes?.ok) createLiveProducts(await liveRes.json());
  } catch (err) {
    console.warn("Auxiliary feed error:", err);
  }
}

// Bootstrap
init();
