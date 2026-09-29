/* ==========================================================================
   CYCLONE SENTINEL · COMMAND CENTER SATELLITE HUD & SIMULATION ENGINE
   Full-screen satellite map, procedural cyclone vortex, and historical replay
   ========================================================================== */

(function () {
  'use strict';

  // --- Global Application State ---
  let map = null;
  let canvas = null;
  let ctx = null;
  let currentStorm = null;
  let availableStorms = [];
  let basinAllStorms = [];

  // Active Layers & Map Tiles
  let activeTileLayer = null;
  let bordersTileLayer = null;
  let tileLayers = {};

  // Vector Feature Groups on Leaflet
  let trackLayerGroup = null;
  let forecastLayerGroup = null;
  let coneLayerGroup = null;
  let radiiLayerGroup = null;
  let allStormsLayerGroup = null;
  let cycloneCenterMarker = null;

  // Layer Visibility Flags
  const layerVisibility = {
    swirl: true,
    track: true,
    cone: true,
    radii: true,
    borders: true,
    allStorms: false,
  };

  // Cloud Simulation Settings
  let cloudMode = 'visible'; // 'visible' | 'infrared' | 'radar'
  let rotationAngle = 0;
  let animationFrameId = null;

  // Timeline & Playback State
  let timelineSequence = []; // Array of { time, lat, lon, wind_kmph, pressure_hpa, stage, isForecast }
  let simIndex = 0; // Float index representing interpolated position along timeline
  let isPlaying = false;
  let playbackSpeed = 1.0;
  let isLooping = true;
  let lastFrameTime = performance.now();

  // Category Color Map (IMD Tropical Cyclone Classification)
  const categoryColors = {
    'Super Cyclonic Storm': '#f43f5e',
    'Extremely Severe Cyclonic Storm': '#ff0055',
    'Very Severe Cyclonic Storm': '#f97316',
    'Severe Cyclonic Storm': '#eab308',
    'Cyclonic Storm': '#10b981',
    'Deep Depression': '#06b6d4',
    'Depression': '#3b82f6',
    'Low Pressure Area': '#64748b',
  };

  function getCategoryColor(stage) {
    if (!stage) return '#38bdf8';
    for (const [key, color] of Object.entries(categoryColors)) {
      if (stage.toLowerCase().includes(key.toLowerCase())) return color;
    }
    return '#38bdf8';
  }

  function number(val, digits = 0) {
    if (val === null || val === undefined || Number.isNaN(Number(val))) return '—';
    return new Intl.NumberFormat('en-IN', { maximumFractionDigits: digits }).format(val);
  }

  function escapeHtml(str) {
    return String(str ?? '').replace(/[&<>'"]/g, char => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[char]));
  }

  function formatTimeUtc(isoString) {
    if (!isoString) return '—';
    return isoString.replace('T', ' ').replace('Z', ' UTC');
  }

  function formatTimeIst(isoString) {
    if (!isoString) return '';
    try {
      const date = new Date(isoString);
      const istTime = date.toLocaleTimeString('en-IN', {
        timeZone: 'Asia/Kolkata',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      });
      return `(${istTime} IST)`;
    } catch {
      return '';
    }
  }

  // ==========================================================================
  // 1. LEAFLET MAP & SATELLITE BASEMAP INITIALIZATION
  // ==========================================================================

  function initMap() {
    const mapHost = document.getElementById('map');
    if (!mapHost) return;

    // Center of North Indian Ocean (Bay of Bengal / Arabian Sea)
    map = L.map('map', {
      center: [14.5, 84.5],
      zoom: 5,
      minZoom: 3,
      maxZoom: 14,
      zoomControl: false, // Using custom HUD zoom buttons
      attributionControl: false,
      worldCopyJump: false,
      maxBounds: L.latLngBounds(L.latLng(-70, -100000), L.latLng(75, 100000)),
      maxBoundsViscosity: 0.85,
    });

    // Basemap Providers (100% Free, High Resolution, No API Key Required)
    tileLayers.satellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, crossOrigin: true }
    );

    // Deep Dark Canvas (No API key needed, high reliability)
    tileLayers.dark = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 16 }
    );

    tileLayers.hybrid = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18 }
    );

    // Reference Overlays (Labels & Boundaries)
    tileLayers.bordersSatellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 18, opacity: 0.85 }
    );

    tileLayers.bordersDark = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 16, opacity: 0.9 }
    );

    // Set default satellite view and boundaries
    activeTileLayer = tileLayers.satellite;
    bordersTileLayer = tileLayers.bordersSatellite;
    activeTileLayer.addTo(map);
    bordersTileLayer.addTo(map);

    // Initialize Vector Feature Groups
    trackLayerGroup = L.layerGroup().addTo(map);
    forecastLayerGroup = L.layerGroup().addTo(map);
    coneLayerGroup = L.layerGroup().addTo(map);
    radiiLayerGroup = L.layerGroup().addTo(map);
    allStormsLayerGroup = L.layerGroup().addTo(map);

    // Setup Canvas for Procedural Cyclone Simulation
    initCanvas();

    // Map Event Listeners
    map.on('move moveend zoom zoomend viewreset resize', () => {
      syncCanvasSize();
      const state = getInterpolatedState();
      if (state && Number.isFinite(state.lat) && Number.isFinite(state.lon)) {
        updateCycloneMarker(state.lat, state.lon, currentStorm?.name || 'Cyclone');

        // Dynamically update vector layers when user scrolls into an adjacent world copy
        const centerLng = map.getCenter().lng;
        const currentK = Math.round((centerLng - state.lon) / 360);
        if (currentK !== lastRenderedK && currentStorm) {
          renderVectorLayers(currentStorm, currentK);
          if (layerVisibility.allStorms) {
            toggleAllStormsLayer(true, currentK);
          }
        }
      }
      renderCycloneCanvas();
    });
  }

  function switchBasemap(mode) {
    if (!map || !tileLayers[mode]) return;
    if (activeTileLayer) map.removeLayer(activeTileLayer);
    if (bordersTileLayer) map.removeLayer(bordersTileLayer);

    activeTileLayer = tileLayers[mode];
    activeTileLayer.addTo(map);

    // Select matching reference/labels layer (dark reference for dark mode, standard for satellite/hybrid)
    bordersTileLayer = (mode === 'dark') ? tileLayers.bordersDark : tileLayers.bordersSatellite;
    if (layerVisibility.borders) {
      bordersTileLayer.addTo(map);
    }
  }

  // ==========================================================================
  // 2. PROCEDURAL CYCLONE SIMULATION CANVAS RENDERER
  // ==========================================================================

  function initCanvas() {
    canvas = document.getElementById('cyclone-canvas');
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    syncCanvasSize();
    window.addEventListener('resize', syncCanvasSize);
    requestAnimationFrame(animationLoop);
  }

  function syncCanvasSize() {
    if (!canvas || !map) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
  }

  // Interpolate current storm state along timeline sequence
  function getInterpolatedState() {
    if (!timelineSequence.length) return null;
    const clampedIndex = Math.max(0, Math.min(timelineSequence.length - 1, simIndex));
    const idx0 = Math.floor(clampedIndex);
    const idx1 = Math.min(timelineSequence.length - 1, idx0 + 1);
    const fraction = clampedIndex - idx0;

    const p0 = timelineSequence[idx0];
    const p1 = timelineSequence[idx1];

    if (!p0) return null;
    if (!p1 || fraction === 0) return { ...p0, index: idx0 };

    return {
      lat: p0.lat + (p1.lat - p0.lat) * fraction,
      lon: p0.lon + (p1.lon - p0.lon) * fraction,
      wind_kmph: Math.round((p0.wind_kmph || 0) + ((p1.wind_kmph || p0.wind_kmph || 0) - (p0.wind_kmph || 0)) * fraction),
      pressure_hpa: (p0.pressure_hpa && p1.pressure_hpa)
        ? Math.round(p0.pressure_hpa + (p1.pressure_hpa - p0.pressure_hpa) * fraction)
        : (p0.pressure_hpa || p1.pressure_hpa || 1000),
      time: fraction > 0.5 ? p1.time : p0.time,
      stage: fraction > 0.5 ? p1.stage : p0.stage,
      isForecast: p0.isForecast || (fraction > 0.5 && p1.isForecast),
      index: idx0,
      fraction,
    };
  }

  function drawProceduralVortex(cx, cy, pixelRadius, eyeRadius, hasDefinedEye, intensity) {
    ctx.save();

    // 1. Atmospheric Low-Pressure Depressive Aura
    const auraGrad = ctx.createRadialGradient(cx, cy, eyeRadius, cx, cy, pixelRadius * 1.25);
    auraGrad.addColorStop(0, 'rgba(0, 0, 0, 0.45)');
    auraGrad.addColorStop(0.7, 'rgba(0, 0, 0, 0.18)');
    auraGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
    ctx.fillStyle = auraGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, pixelRadius * 1.25, 0, Math.PI * 2);
    ctx.fill();

    // 2. Swirling Logarithmic Spiral Cloud Bands (Northern Hemisphere: Counter-Clockwise Inflow)
    const numArms = 5;
    const steps = 32;

    for (let arm = 0; arm < numArms; arm++) {
      const armBaseAngle = (arm * Math.PI * 2) / numArms + rotationAngle;

      for (let s = 0; s < steps; s++) {
        const t = s / steps; // 0 (outer) to 1 (near eye)
        // Logarithmic spiral inwards: radius decreases, angle wraps counter-clockwise
        const r = eyeRadius + (pixelRadius - eyeRadius) * Math.pow(1 - t, 1.35);
        const theta = armBaseAngle - 2.8 * Math.pow(t, 0.75);

        const px = cx + Math.cos(theta) * r;
        const py = cy + Math.sin(theta) * r;

        // Size of convective cloud puff
        const puffRadius = Math.max(10, (pixelRadius * 0.14) * (0.5 + t * 0.8));

        // Soft cloud gradient depending on sensor mode
        const puffGrad = ctx.createRadialGradient(px, py, 0, px, py, puffRadius);

        if (cloudMode === 'infrared') {
          // Cold convective cloud tops (Cyan -> Blue -> Yellow -> Red -> Violet)
          if (t > 0.75) {
            // Coldest tops around eyewall (-75°C)
            puffGrad.addColorStop(0, 'rgba(236, 72, 153, 0.85)');
            puffGrad.addColorStop(0.4, 'rgba(239, 68, 68, 0.75)');
            puffGrad.addColorStop(0.8, 'rgba(234, 179, 8, 0.4)');
            puffGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
          } else if (t > 0.4) {
            // Moderate convection (-50°C)
            puffGrad.addColorStop(0, 'rgba(234, 179, 8, 0.8)');
            puffGrad.addColorStop(0.5, 'rgba(6, 182, 212, 0.6)');
            puffGrad.addColorStop(1, 'rgba(59, 130, 246, 0)');
          } else {
            // Outer cirrus (-30°C)
            puffGrad.addColorStop(0, 'rgba(56, 189, 248, 0.6)');
            puffGrad.addColorStop(0.6, 'rgba(30, 58, 138, 0.35)');
            puffGrad.addColorStop(1, 'rgba(15, 23, 42, 0)');
          }
        } else if (cloudMode === 'radar') {
          // Doppler Radar Reflectivity
          if (t > 0.75) {
            puffGrad.addColorStop(0, 'rgba(239, 68, 68, 0.9)'); // High dBZ
            puffGrad.addColorStop(0.6, 'rgba(249, 115, 22, 0.7)');
            puffGrad.addColorStop(1, 'rgba(234, 179, 8, 0)');
          } else if (t > 0.4) {
            puffGrad.addColorStop(0, 'rgba(234, 179, 8, 0.8)');
            puffGrad.addColorStop(0.6, 'rgba(16, 185, 129, 0.6)');
            puffGrad.addColorStop(1, 'rgba(6, 182, 212, 0)');
          } else {
            puffGrad.addColorStop(0, 'rgba(16, 185, 129, 0.6)');
            puffGrad.addColorStop(0.7, 'rgba(56, 189, 248, 0.3)');
            puffGrad.addColorStop(1, 'rgba(15, 23, 42, 0)');
          }
        } else {
          // Visible Satellite (Natural Cloud Texture with subtle 3D ocean shadow)
          const coreOpacity = 0.55 + t * 0.4;
          puffGrad.addColorStop(0, `rgba(255, 255, 255, ${coreOpacity})`);
          puffGrad.addColorStop(0.45, `rgba(240, 246, 252, ${coreOpacity * 0.65})`);
          puffGrad.addColorStop(0.85, `rgba(210, 225, 240, ${coreOpacity * 0.25})`);
          puffGrad.addColorStop(1, 'rgba(180, 205, 225, 0)');
        }

        ctx.fillStyle = puffGrad;
        ctx.beginPath();
        ctx.arc(px, py, puffRadius, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 3. Central Dense Overcast (CDO) Shield
    const cdoRadius = Math.max(30, pixelRadius * 0.38);
    const cdoGrad = ctx.createRadialGradient(cx, cy, eyeRadius, cx, cy, cdoRadius);
    if (cloudMode === 'infrared') {
      cdoGrad.addColorStop(0, 'rgba(244, 63, 94, 0.95)');
      cdoGrad.addColorStop(0.6, 'rgba(234, 179, 8, 0.8)');
      cdoGrad.addColorStop(1, 'rgba(56, 189, 248, 0)');
    } else if (cloudMode === 'radar') {
      cdoGrad.addColorStop(0, 'rgba(239, 68, 68, 0.95)');
      cdoGrad.addColorStop(0.6, 'rgba(249, 115, 22, 0.8)');
      cdoGrad.addColorStop(1, 'rgba(16, 185, 129, 0)');
    } else {
      cdoGrad.addColorStop(0, 'rgba(255, 255, 255, 0.95)');
      cdoGrad.addColorStop(0.5, 'rgba(245, 250, 255, 0.85)');
      cdoGrad.addColorStop(1, 'rgba(215, 230, 245, 0)');
    }
    ctx.fillStyle = cdoGrad;
    ctx.beginPath();
    ctx.arc(cx, cy, cdoRadius, 0, Math.PI * 2);
    ctx.fill();

    // 4. Distinct Eye Cavity & Eye Wall (Clear ocean center for strong cyclones)
    if (hasDefinedEye) {
      // Clear out center eye with destination-out feathering
      ctx.save();
      ctx.globalCompositeOperation = 'destination-out';
      const eyeHoleGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, eyeRadius);
      eyeHoleGrad.addColorStop(0, 'rgba(0, 0, 0, 0.92)');
      eyeHoleGrad.addColorStop(0.7, 'rgba(0, 0, 0, 0.7)');
      eyeHoleGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = eyeHoleGrad;
      ctx.beginPath();
      ctx.arc(cx, cy, eyeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Sharp Dense Eyewall Ring
      ctx.strokeStyle = cloudMode === 'infrared' ? '#f43f5e' : (cloudMode === 'radar' ? '#ef4444' : '#ffffff');
      ctx.lineWidth = Math.max(2.5, eyeRadius * 0.22);
      ctx.beginPath();
      ctx.arc(cx, cy, eyeRadius, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  function renderCycloneCanvas() {
    if (!ctx || !canvas || !map) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    ctx.clearRect(0, 0, width, height);

    if (!layerVisibility.swirl || !timelineSequence.length) return;

    const state = getInterpolatedState();
    if (!state || !Number.isFinite(state.lat) || !Number.isFinite(state.lon)) return;

    // Calculate physical radius in screen pixels based on map zoom
    const zoom = map.getZoom();
    const windSpeed = state.wind_kmph || 60;
    const intensity = Math.min(1.0, Math.max(0.15, (windSpeed - 30) / 190));

    // Base storm cloud radius in km (approx 240 - 450 km)
    const stormRadiusKm = 240 + intensity * 180;
    // Conversion factor km to pixels at current latitude and zoom
    const kmPerPixel = (40075 * Math.cos(state.lat * Math.PI / 180)) / Math.pow(2, zoom + 8);
    const pixelRadius = Math.max(70, Math.min(750, stormRadiusKm / kmPerPixel));

    // Eye radius in pixels
    const hasDefinedEye = windSpeed >= 90;
    const eyeRadius = Math.max(6, Math.min(42, (hasDefinedEye ? 14 + intensity * 16 : 8) / (kmPerPixel * 0.4)));

    // Calculate dynamic offsets relative to current viewport center longitude
    // Guarantees procedural swirl renders on any world copy the user pans to
    const centerLng = map.getCenter().lng;
    const baseK = Math.round((centerLng - state.lon) / 360);
    const offsets = [
      (baseK - 1) * 360,
      baseK * 360,
      (baseK + 1) * 360
    ];

    for (const offset of offsets) {
      const centerPoint = map.latLngToContainerPoint([state.lat, state.lon + offset]);
      const cx = centerPoint.x;
      const cy = centerPoint.y;

      // Only render if visible on or near screen
      if (cx + pixelRadius < -100 || cx - pixelRadius > width + 100 ||
          cy + pixelRadius < -100 || cy - pixelRadius > height + 100) {
        continue;
      }

      drawProceduralVortex(cx, cy, pixelRadius, eyeRadius, hasDefinedEye, intensity);
    }
  }

  function animationLoop(timestamp) {
    const deltaMs = timestamp - lastFrameTime;
    lastFrameTime = timestamp;

    // Advance continuous counter-clockwise cloud swirl rotation
    const baseSpeed = 0.016;
    rotationAngle -= baseSpeed * playbackSpeed;

    // Advance timeline simulation if playing
    if (isPlaying && timelineSequence.length > 1) {
      const stepAdvance = (deltaMs / 1000) * (0.45 * playbackSpeed);
      simIndex += stepAdvance;

      if (simIndex >= timelineSequence.length - 1) {
        if (isLooping) {
          simIndex = 0;
        } else {
          simIndex = timelineSequence.length - 1;
          pauseSimulation();
        }
      }

      updateSimulationUI();
    }

    renderCycloneCanvas();
    animationFrameId = requestAnimationFrame(animationLoop);
  }

  // ==========================================================================
  // 3. TIMELINE & SIMULATION CONTROLLER ENGINE
  // ==========================================================================

  function buildTimelineSequence(storm) {
    timelineSequence = [];
    const observed = storm.observed_track || [];
    const forecast = storm.forecast_track || [];

    // Add all observed track points
    observed.forEach(pt => {
      timelineSequence.push({
        time: pt.time,
        lat: Number(pt.lat),
        lon: Number(pt.lon),
        wind_kmph: pt.wind_kmph,
        pressure_hpa: pt.pressure_hpa,
        stage: pt.stage,
        isForecast: false,
      });
    });

    // Append baseline forecast points (+24h, +48h, +72h)
    forecast.forEach(pt => {
      timelineSequence.push({
        time: pt.time,
        lat: Number(pt.lat),
        lon: Number(pt.lon),
        wind_kmph: pt.wind_kmph,
        pressure_hpa: null,
        stage: `${pt.hours}h Baseline Forecast`,
        isForecast: true,
      });
    });

    // Default to the latest recorded observed point
    const lastObservedIndex = Math.max(0, observed.length - 1);
    simIndex = lastObservedIndex;

    // Update Slider Bounds
    const slider = document.getElementById('timeline-slider');
    if (slider) {
      slider.min = '0';
      slider.max = String(Math.max(1, timelineSequence.length - 1));
      slider.step = '0.05';
      slider.value = String(simIndex);
    }

    updateSimulationUI();
  }

  function updateSimulationUI() {
    const state = getInterpolatedState();
    if (!state) return;

    // 1. Update Slider Element & Visual Thumb Position
    const slider = document.getElementById('timeline-slider');
    if (slider) {
      slider.value = String(simIndex);
      const pct = (simIndex / Math.max(1, timelineSequence.length - 1)) * 100;
      document.querySelector('.bottom-hud')?.style.setProperty('--slider-pct', `${pct}%`);
    }

    // 2. Update Bottom HUD Readouts
    const stepBadge = document.getElementById('sim-step-badge');
    const phaseBadge = document.getElementById('sim-phase-badge');
    const timeUtc = document.getElementById('sim-time-utc');
    const timeIst = document.getElementById('sim-time-ist');
    const pill = document.getElementById('sim-telemetry-pill');

    if (stepBadge) {
      stepBadge.textContent = `OBSERVATION ${state.index + 1} / ${timelineSequence.length}`;
    }

    if (phaseBadge) {
      if (state.isForecast) {
        phaseBadge.textContent = 'BASELINE FORECAST';
        phaseBadge.className = 'phase-badge phase-forecast';
      } else {
        phaseBadge.textContent = 'HISTORICAL REPLAY';
        phaseBadge.className = 'phase-badge phase-observed';
      }
    }

    if (timeUtc) timeUtc.textContent = formatTimeUtc(state.time);
    if (timeIst) timeIst.textContent = formatTimeIst(state.time);

    if (pill) {
      pill.textContent = `${number(state.wind_kmph)} km/h · ${state.pressure_hpa ? number(state.pressure_hpa) + ' hPa · ' : ''}${state.stage || 'Storm'}`;
    }

    // 3. Update Top Header Ticker
    const tickerCoords = document.getElementById('hud-storm-coords');
    const tickerWind = document.getElementById('hud-storm-wind');
    const tickerPres = document.getElementById('hud-storm-pressure');
    const tickerCat = document.getElementById('hud-storm-category');

    if (tickerCoords) tickerCoords.textContent = `${number(state.lat, 2)}°N, ${number(state.lon, 2)}°E`;
    if (tickerWind) tickerWind.innerHTML = `💨 <strong>${number(state.wind_kmph)}</strong> km/h`;
    if (tickerPres) tickerPres.innerHTML = `⏱ <strong>${number(state.pressure_hpa)}</strong> hPa`;
    if (tickerCat && state.stage) {
      tickerCat.textContent = state.stage;
      tickerCat.style.color = getCategoryColor(state.stage);
      tickerCat.style.borderColor = getCategoryColor(state.stage);
    }

    // 4. Update Gauges in Left Sidebar
    const windVal = document.getElementById('telemetry-wind-val');
    const windKts = document.getElementById('telemetry-wind-kts');
    const windFill = document.getElementById('wind-speed-fill');
    const presVal = document.getElementById('telemetry-pressure-val');
    const presFill = document.getElementById('pressure-fill');

    if (windVal) windVal.textContent = number(state.wind_kmph);
    if (windKts) windKts.textContent = `${number(Math.round(state.wind_kmph / 1.852))} knots`;
    if (windFill) {
      const pct = Math.min(100, Math.max(10, ((state.wind_kmph || 0) / 260) * 100));
      windFill.style.width = `${pct}%`;
    }

    if (presVal) presVal.textContent = number(state.pressure_hpa);
    if (presFill && state.pressure_hpa) {
      const dropPct = Math.min(100, Math.max(5, ((1010 - state.pressure_hpa) / 100) * 100));
      presFill.style.width = `${dropPct}%`;
    }

    // 5. Update Center Map Marker Position
    updateCycloneMarker(state.lat, state.lon, currentStorm?.name || 'Cyclone');
  }

  function playSimulation() {
    isPlaying = true;
    const playBtn = document.getElementById('btn-sim-play');
    const playIcon = document.getElementById('play-icon');
    if (playBtn) playBtn.classList.add('playing');
    if (playIcon) playIcon.textContent = '⏸';
  }

  function pauseSimulation() {
    isPlaying = false;
    const playBtn = document.getElementById('btn-sim-play');
    const playIcon = document.getElementById('play-icon');
    if (playBtn) playBtn.classList.remove('playing');
    if (playIcon) playIcon.textContent = '▶';
  }

  function togglePlaySimulation() {
    if (isPlaying) {
      pauseSimulation();
    } else {
      if (simIndex >= timelineSequence.length - 1) simIndex = 0;
      playSimulation();
    }
  }

  // ==========================================================================
  // 4. VECTOR LAYERS: TRACKS, WAYPOINTS, CONE & ASYMMETRIC RADII
  // ==========================================================================

  let lastRenderedK = 0;
  let lastAllStormsK = 0;

  function updateCycloneMarker(lat, lon, label) {
    if (!map || !Number.isFinite(lat) || !Number.isFinite(lon)) return;

    // Anchor cyclone center HUD marker to the world copy closest to the viewport center
    const centerLng = map.getCenter().lng;
    const wrappedLon = lon + Math.round((centerLng - lon) / 360) * 360;

    if (!cycloneCenterMarker) {
      const icon = L.divIcon({
        className: 'cyclone-center-marker',
        html: `
          <div class="center-marker-core"></div>
          <div class="center-marker-ring"></div>
          <div class="center-marker-label" id="marker-storm-label">${escapeHtml(label)}</div>
        `,
        iconSize: [20, 20],
        iconAnchor: [10, 10],
      });
      cycloneCenterMarker = L.marker([lat, wrappedLon], { icon, zIndexOffset: 1000 }).addTo(map);
    } else {
      cycloneCenterMarker.setLatLng([lat, wrappedLon]);
      const labelEl = document.getElementById('marker-storm-label');
      if (labelEl) labelEl.textContent = label;
    }
  }

  function renderVectorLayers(storm, centerK = null) {
    if (!map || !storm) return;

    const observed = storm.observed_track || [];
    const forecast = storm.forecast_track || [];
    const current = storm.current || {};

    const baseLon = (observed[0]?.lon) || (current?.lon) || 84.5;
    const centerLng = map.getCenter().lng;
    const k = centerK !== null ? centerK : Math.round((centerLng - baseLon) / 360);
    lastRenderedK = k;

    // Clear previous vector layers
    trackLayerGroup.clearLayers();
    forecastLayerGroup.clearLayers();
    coneLayerGroup.clearLayers();
    radiiLayerGroup.clearLayers();

    // Render on the current world copy, as well as adjacent copies
    const offsets = [
      (k - 1) * 360,
      k * 360,
      (k + 1) * 360
    ];

    offsets.forEach(offset => {
      // 1. Observed Best-Track Polyline & Waypoints
      if (layerVisibility.track && observed.length > 0) {
        const latLngs = observed.map(pt => [pt.lat, pt.lon + offset]);

        // Glowing Main Track Line (noClip: true prevents Leaflet SVG viewport culling)
        L.polyline(latLngs, {
          color: '#10b981',
          weight: 3.5,
          opacity: 0.9,
          noClip: true,
        }).addTo(trackLayerGroup);

        // Waypoint Dots
        observed.forEach((pt, idx) => {
          const isLatest = idx === observed.length - 1;
          const radius = isLatest ? 6 : 4;
          const color = getCategoryColor(pt.stage);

          const circle = L.circleMarker([pt.lat, pt.lon + offset], {
            radius,
            fillColor: color,
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 1,
          }).addTo(trackLayerGroup);

          // Ensure Leaflet SVG renderer does not cull circle markers
          circle._empty = () => false;

          circle.bindTooltip(`
            <div style="font-family: var(--font-main); font-size: 11px;">
              <strong style="color: ${color};">${escapeHtml(storm.name)}</strong><br/>
              <span>${formatTimeUtc(pt.time)}</span><br/>
              <span>Wind: <b>${number(pt.wind_kmph)} km/h</b> · ${number(pt.pressure_hpa)} hPa</span><br/>
              <small style="color: #94a9c4;">${escapeHtml(pt.stage)}</small>
            </div>
          `, { sticky: true, opacity: 0.95 });

          circle.on('click', () => {
            pauseSimulation();
            simIndex = idx;
            updateSimulationUI();
          });
        });
      }

      // 2. Forecast Baseline Polyline
      if (layerVisibility.track && forecast.length > 0 && observed.length > 0) {
        const lastObs = observed[observed.length - 1];
        const forecastLatLngs = [
          [lastObs.lat, lastObs.lon + offset],
          ...forecast.map(pt => [pt.lat, pt.lon + offset])
        ];

        L.polyline(forecastLatLngs, {
          color: '#38bdf8',
          weight: 3,
          dashArray: '6, 6',
          opacity: 0.85,
          noClip: true,
        }).addTo(forecastLayerGroup);

        forecast.forEach((pt, idx) => {
          const circle = L.circleMarker([pt.lat, pt.lon + offset], {
            radius: 5,
            fillColor: '#38bdf8',
            color: '#ffffff',
            weight: 1.5,
            fillOpacity: 1,
          }).addTo(forecastLayerGroup);

          circle._empty = () => false;

          circle.bindTooltip(`
            <div style="font-family: var(--font-main); font-size: 11px;">
              <strong style="color: #38bdf8;">+${pt.hours}h Research Baseline</strong><br/>
              <span>${formatTimeUtc(pt.time)}</span><br/>
              <span>Wind: <b>${number(pt.wind_kmph)} km/h</b></span><br/>
              <span>Uncertainty Radius: ±${number(pt.radius_km)} km</span>
            </div>
          `, { sticky: true });

          circle.on('click', () => {
            pauseSimulation();
            simIndex = observed.length + idx;
            updateSimulationUI();
          });
        });
      }

      // 3. Held-out Error Uncertainty Cone Polygon
      if (layerVisibility.cone && (storm.cone_polygon || []).length > 2) {
        const offsetCone = storm.cone_polygon.map(([lat, lon]) => [lat, lon + offset]);
        L.polygon(offsetCone, {
          color: '#38bdf8',
          weight: 1.5,
          dashArray: '4, 4',
          fillColor: '#38bdf8',
          fillOpacity: 0.12,
          noClip: true,
        }).addTo(coneLayerGroup);
      }

      // 4. Asymmetric Wind Radii Extents
      if (layerVisibility.radii && current && Number.isFinite(current.lat) && Number.isFinite(current.lon)) {
        const radii = current.wind_radii || {};

        const drawQuadrantRings = (quadrants, color, label) => {
          if (!quadrants) return;
          const avgKm = (quadrants.ne + quadrants.se + quadrants.sw + quadrants.nw) / 4;
          if (!avgKm || avgKm <= 0) return;

          const ring = L.circle([current.lat, current.lon + offset], {
            radius: avgKm * 1000,
            color,
            weight: 1.5,
            fillColor: color,
            fillOpacity: 0.08,
          });
          ring._empty = () => false;
          ring.bindTooltip(`${label}: ~${Math.round(avgKm)} km extent`, { sticky: true }).addTo(radiiLayerGroup);
        };

        drawQuadrantRings(radii.r34_km, '#facc15', '34 kt Gale Wind');
        drawQuadrantRings(radii.r50_km, '#fb923c', '50 kt Storm Wind');
        drawQuadrantRings(radii.r64_km, '#f87171', '64 kt Hurricane Wind');
      }
    });
  }

  // ==========================================================================
  // 5. BASIN OVERVIEW: RENDER ALL HISTORICAL CYCLONE TRACKS
  // ==========================================================================

  async function loadBasinAllStorms() {
    if (basinAllStorms.length > 0) return;
    try {
      const res = await fetch('/api/v1/storms/basin/all');
      if (res.ok) {
        basinAllStorms = await res.json();
      }
    } catch {
      basinAllStorms = [];
    }
  }

  function toggleAllStormsLayer(show, centerK = null) {
    allStormsLayerGroup.clearLayers();

    if (!show) return;

    const centerLng = map ? map.getCenter().lng : 84.5;
    const k = centerK !== null ? centerK : Math.round((centerLng - 84.5) / 360);
    lastAllStormsK = k;

    const offsets = [
      (k - 1) * 360,
      k * 360,
      (k + 1) * 360
    ];

    offsets.forEach(offset => {
      basinAllStorms.forEach(s => {
        if (!s.points || s.points.length < 2) return;
        const isSelected = s.id === currentStorm?.id;
        const color = isSelected ? '#ffffff' : getCategoryColor(s.peak_category);

        const offsetPoints = s.points.map(([lat, lon]) => [lat, lon + offset]);
        const line = L.polyline(offsetPoints, {
          color,
          weight: isSelected ? 4 : 2,
          opacity: isSelected ? 1 : 0.45,
          noClip: true,
        }).addTo(allStormsLayerGroup);

        line.bindTooltip(`
          <div style="font-family: var(--font-main); font-size: 11px;">
            <strong style="color: ${color};">${escapeHtml(s.name)} (${escapeHtml(s.season)})</strong><br/>
            <span>Peak: <b>${escapeHtml(s.peak_category)}</b></span><br/>
            <span>Max Wind: ${number(s.peak_wind_kmph)} km/h</span><br/>
            <small style="color: var(--accent-cyan);">Click to simulate</small>
          </div>
        `, { sticky: true });

        line.on('click', () => {
          loadStorm(s.id);
        });
      });
    });
  }

  // ==========================================================================
  // 6. POPULATE SIDEBAR & HUD DATA TABS
  // ==========================================================================

  function renderSidebarTelemetry(storm) {
    const current = storm.current || {};
    const dvorak = storm.dvorak || {};
    const ri = storm.rapid_intensification || {};

    // Hero title & status
    const heroTitle = document.getElementById('telemetry-storm-title');
    const heroSub = document.getElementById('telemetry-last-observed');
    const heroBadge = document.getElementById('telemetry-category-badge');
    const heroSeason = document.getElementById('telemetry-season');

    if (heroTitle) heroTitle.textContent = storm.name || 'Cyclone';
    if (heroSub) heroSub.textContent = `Last observation: ${formatTimeUtc(storm.last_updated)}`;
    if (heroSeason) heroSeason.textContent = `Season ${storm.season || '—'}`;
    if (heroBadge) {
      heroBadge.textContent = storm.status || 'Cyclonic Storm';
      heroBadge.style.color = getCategoryColor(storm.status);
    }

    // Mini cards
    const dvorakVal = document.getElementById('telemetry-dvorak-val');
    const dvorakPat = document.getElementById('telemetry-dvorak-pattern');
    if (dvorakVal) dvorakVal.textContent = dvorak.t_number ? `T${number(dvorak.t_number, 1)}` : '—';
    if (dvorakPat) dvorakPat.textContent = dvorak.pattern_type || 'Intensity-derived proxy';

    const riScore = document.getElementById('telemetry-ri-score');
    const riBadge = document.getElementById('telemetry-ri-badge');
    if (riScore) riScore.textContent = ri.ri_score !== undefined ? `${Math.round(ri.ri_score * 100)}/100` : '—';
    if (riBadge) riBadge.textContent = ri.status_label || 'Screening';

    const rainVal = document.getElementById('telemetry-rainfall-val');
    const rainSrc = document.getElementById('telemetry-rainfall-source');
    if (rainVal) rainVal.textContent = current.rainfall_mm_hr !== null && current.rainfall_mm_hr !== undefined ? `${number(current.rainfall_mm_hr, 1)} mm/h` : 'No feature';
    if (rainSrc) rainSrc.textContent = current.rainfall_mm_hr !== null ? 'GPM IMERG Final' : 'Unmatched feature';

    // Wind Radii Readout
    const radii = current.wind_radii || {};
    const r34El = document.getElementById('radii-r34-val');
    const r50El = document.getElementById('radii-r50-val');
    const r64El = document.getElementById('radii-r64-val');

    const fmtQuad = q => q ? `NE:${q.ne} SE:${q.se} SW:${q.sw} NW:${q.nw} km` : 'Threshold not met';
    if (r34El) r34El.textContent = fmtQuad(radii.r34_km);
    if (r50El) r50El.textContent = fmtQuad(radii.r50_km);
    if (r64El) r64El.textContent = fmtQuad(radii.r64_km);
  }

  function renderSidebarAnalysis(storm) {
    const dvorak = storm.dvorak || {};
    const ri = storm.rapid_intensification || {};
    const metrics = storm.model_metrics?.test_metrics || {};

    // Dvorak
    const tEl = document.getElementById('analysis-dvorak-t');
    const typeEl = document.getElementById('analysis-dvorak-type');
    const ciEl = document.getElementById('analysis-dvorak-ci');
    const descEl = document.getElementById('analysis-dvorak-desc');
    const defEl = document.getElementById('analysis-dvorak-deficit');
    const envEl = document.getElementById('analysis-dvorak-env-pres');
    const eyeEl = document.getElementById('analysis-dvorak-eye');

    if (tEl) tEl.textContent = dvorak.t_number ? `T${number(dvorak.t_number, 1)}` : '—';
    if (typeEl) typeEl.textContent = dvorak.pattern_type || 'Analysis mode';
    if (ciEl) ciEl.textContent = `CI ${number(dvorak.ci_number, 1) || '—'}`;
    if (descEl) descEl.textContent = dvorak.pattern_description || 'Intensity-derived Dvorak proxy without imagery analysis.';
    if (defEl) defEl.textContent = `${number(dvorak.central_pressure_deficit_hpa, 1)} hPa`;
    if (envEl) envEl.textContent = `${number(dvorak.environmental_pressure_hpa, 1)} hPa`;
    if (eyeEl) eyeEl.textContent = dvorak.eye_characteristics?.eye_type || 'Central Convection';

    // RI screening
    const riNum = document.getElementById('analysis-ri-num');
    const riHeading = document.getElementById('analysis-ri-heading');
    const riSummary = document.getElementById('analysis-ri-summary');
    const riTag = document.getElementById('analysis-ri-status-tag');
    const factorsList = document.getElementById('analysis-ri-factors');

    const score = Math.round((ri.ri_score || 0) * 100);
    if (riNum) riNum.textContent = String(score);
    if (riHeading) riHeading.textContent = ri.status_label || 'RI Screen';
    if (riSummary) riSummary.textContent = ri.summary || 'Heuristic screening based on seasonal baseline.';
    if (riTag) {
      riTag.textContent = ri.status_label || 'Screening';
      riTag.style.color = score > 65 ? '#f43f5e' : (score > 35 ? '#f59e0b' : '#10b981');
    }

    if (factorsList) {
      const factors = Object.values(ri.factor_scores || {});
      factorsList.innerHTML = factors.map(f => `
        <div class="factor-item">
          <span>${escapeHtml(f.name)}</span>
          <b class="${f.favorable ? 'factor-favorable' : 'factor-limiting'}">
            ${escapeHtml(f.value)} (${f.favorable ? 'favourable' : 'limiting'})
          </b>
        </div>
      `).join('');
    }

    // Model test set errors
    const err24 = document.getElementById('analysis-err-24');
    const err48 = document.getElementById('analysis-err-48');
    const err72 = document.getElementById('analysis-err-72');

    const baseMae = metrics.track?.endpoint_mae_km || 150;
    if (err24) err24.textContent = `±${Math.round(baseMae)} km`;
    if (err48) err48.textContent = `±${Math.round(baseMae * Math.sqrt(2))} km`;
    if (err72) err72.textContent = `±${Math.round(baseMae * Math.sqrt(3))} km`;
  }

  function renderSidebarCoastal(storm) {
    const landfall = storm.landfall || {};
    const districts = storm.district_risk || [];

    const lmEl = document.getElementById('coastal-nearest-landmark');
    const portEl = document.getElementById('coastal-nearest-port');
    const etaEl = document.getElementById('coastal-eta-ist');
    const surgeEl = document.getElementById('coastal-surge-proxy');
    const listEl = document.getElementById('coastal-district-list');

    if (lmEl) lmEl.textContent = landfall.nearest_landmark || 'Open ocean';
    if (portEl) portEl.textContent = landfall.nearest_port || '—';
    if (etaEl) etaEl.textContent = landfall.eta_timestamp_ist || 'No proximity flag';
    if (surgeEl) surgeEl.textContent = landfall.storm_surge_meters ? `${number(landfall.storm_surge_meters, 1)} m proxy` : 'Not evaluated';

    if (listEl) {
      if (!districts.length) {
        listEl.innerHTML = '<p class="empty-list-notice">No reference districts within proximity radius of research track.</p>';
      } else {
        listEl.innerHTML = districts.map(d => `
          <div class="district-row">
            <div>
              <strong>${escapeHtml(d.district)}</strong>
              <small>${number(d.distance_to_track_km)} km from baseline · Exposed pop: ${number(d.population_exposed)}</small>
            </div>
            <span class="district-tier">${escapeHtml(d.alert_tier)}</span>
          </div>
        `).join('');
      }
    }
  }

  async function renderSidebarSources() {
    const pipeList = document.getElementById('sources-pipeline-list');
    const pipeNext = document.getElementById('sources-pipeline-next');
    const liveList = document.getElementById('sources-live-list');
    const liveNext = document.getElementById('sources-live-next');

    try {
      const [catRes, liveRes] = await Promise.all([
        fetch('/api/v1/data/catalog'),
        fetch('/api/v1/live/products')
      ]);

      const cat = await catRes.json();
      const live = await liveRes.json();

      if (pipeList && cat.layers) {
        pipeList.innerHTML = cat.layers.map(l => {
          const st = l.status || 'pending';
          return `
            <div class="source-item">
              <div>
                <h4>${escapeHtml(l.name)}</h4>
                <p>${escapeHtml(l.purpose || '')}</p>
              </div>
              <span class="source-badge source-${st}">${escapeHtml(st.replace(/_/g, ' '))}</span>
            </div>
          `;
        }).join('');
      }

      if (pipeNext) pipeNext.textContent = cat.next_milestone || '';

      if (liveList && live.products) {
        liveList.innerHTML = live.products.map(p => {
          const st = p.status || 'not_polled';
          return `
            <div class="source-item">
              <div>
                <h4>${escapeHtml(p.name)}</h4>
                <p>${escapeHtml(p.provider || '')} · ${escapeHtml(p.mode || '')}</p>
              </div>
              <span class="source-badge source-${st}">${escapeHtml(st.replace(/_/g, ' '))}</span>
            </div>
          `;
        }).join('');
      }

      if (liveNext) liveNext.textContent = live.next_step || '';
    } catch {
      if (pipeNext) pipeNext.textContent = 'Data source readiness could not be retrieved.';
    }
  }

  // ==========================================================================
  // 7. STORM LOADING & SEARCHABLE SELECTOR
  // ==========================================================================

  async function loadStormList(selectedId) {
    const res = await fetch('/api/v1/storms');
    if (!res.ok) throw new Error('Could not fetch collected cyclone list.');
    availableStorms = await res.json();
    renderStormDropdownOptions();
    updateStormSelectLabel(selectedId);
  }

  function updateStormSelectLabel(stormId) {
    const nameEl = document.getElementById('storm-select-name');
    const metaEl = document.getElementById('storm-select-meta');
    const tickerName = document.getElementById('hud-storm-name');
    const found = availableStorms.find(s => s.id === stormId) || currentStorm;

    if (found) {
      if (nameEl) nameEl.textContent = found.name;
      if (metaEl) metaEl.textContent = `${found.season} · ${found.peak_category || 'Cyclonic Storm'}`;
      if (tickerName) tickerName.textContent = found.name;
    }
  }

  function renderStormDropdownOptions(query = '', filter = 'all') {
    const container = document.getElementById('storm-options-list');
    if (!container) return;

    const q = query.trim().toLowerCase();
    const filtered = availableStorms.filter(s => {
      const matchText = `${s.name} ${s.season} ${s.peak_category}`.toLowerCase().includes(q);
      if (!matchText) return false;

      if (filter === 'recent') return Number(s.season) >= 2020;
      if (filter !== 'all') return (s.peak_category || '').toLowerCase().includes(filter.toLowerCase());
      return true;
    });

    if (!filtered.length) {
      container.innerHTML = '<p style="padding: 12px; color: var(--text-muted); font-size: 11px; text-align: center;">No cyclones matched your search.</p>';
      return;
    }

    container.innerHTML = filtered.map(s => `
      <button class="storm-option-row" type="button" data-storm-id="${escapeHtml(s.id)}" aria-selected="${s.id === currentStorm?.id}">
        <div>
          <span class="storm-option-name">${escapeHtml(s.name)}</span>
          <span class="storm-option-cat">${escapeHtml(s.peak_category || 'Cyclonic Storm')}</span>
        </div>
        <span class="storm-option-season">${escapeHtml(s.season)}</span>
      </button>
    `).join('');

    container.querySelectorAll('.storm-option-row').forEach(row => {
      row.onclick = () => {
        closeStormDropdown();
        loadStorm(row.dataset.stormId);
      };
    });
  }

  function closeStormDropdown() {
    const menu = document.getElementById('storm-dropdown-menu');
    const trigger = document.getElementById('storm-select-button');
    if (menu) menu.hidden = true;
    if (trigger) trigger.setAttribute('aria-expanded', 'false');
  }

  function setupStormSelector() {
    const trigger = document.getElementById('storm-select-button');
    const menu = document.getElementById('storm-dropdown-menu');
    const searchInput = document.getElementById('storm-search-input');
    const prevBtn = document.getElementById('btn-prev-storm');
    const nextBtn = document.getElementById('btn-next-storm');

    if (trigger && menu) {
      trigger.onclick = () => {
        const isOpen = !menu.hidden;
        menu.hidden = isOpen;
        trigger.setAttribute('aria-expanded', String(!isOpen));
        if (!isOpen && searchInput) {
          searchInput.focus();
        }
      };

      document.addEventListener('click', e => {
        if (!e.target.closest('.storm-selector-panel')) closeStormDropdown();
      });
    }

    if (searchInput) {
      searchInput.oninput = () => {
        const activeChip = document.querySelector('.filter-chips .chip.active');
        const filter = activeChip?.dataset.filter || 'all';
        renderStormDropdownOptions(searchInput.value, filter);
      };
    }

    document.querySelectorAll('.filter-chips .chip').forEach(chip => {
      chip.onclick = () => {
        document.querySelectorAll('.filter-chips .chip').forEach(c => c.classList.remove('active'));
        chip.classList.add('active');
        const query = searchInput ? searchInput.value : '';
        renderStormDropdownOptions(query, chip.dataset.filter);
      };
    });

    // Cycle Previous / Next Storm Buttons
    if (prevBtn) {
      prevBtn.onclick = () => {
        if (!availableStorms.length || !currentStorm) return;
        const idx = availableStorms.findIndex(s => s.id === currentStorm.id);
        const prevIdx = (idx - 1 + availableStorms.length) % availableStorms.length;
        loadStorm(availableStorms[prevIdx].id);
      };
    }

    if (nextBtn) {
      nextBtn.onclick = () => {
        if (!availableStorms.length || !currentStorm) return;
        const idx = availableStorms.findIndex(s => s.id === currentStorm.id);
        const nextIdx = (idx + 1) % availableStorms.length;
        loadStorm(availableStorms[nextIdx].id);
      };
    }
  }

  async function loadStorm(stormId) {
    pauseSimulation();
    try {
      const res = await fetch(`/api/v1/storms/${encodeURIComponent(stormId)}`);
      if (!res.ok) throw new Error('Historical cyclone replay not found.');
      currentStorm = await res.json();

      updateStormSelectLabel(currentStorm.id);
      buildTimelineSequence(currentStorm);
      renderSidebarTelemetry(currentStorm);
      renderSidebarAnalysis(currentStorm);
      renderSidebarCoastal(currentStorm);
      renderVectorLayers(currentStorm);

      // Smooth pan camera to cyclone genesis or latest location
      const observed = currentStorm.observed_track || [];
      const targetPoint = observed[observed.length - 1] || currentStorm.current;
      if (targetPoint && Number.isFinite(targetPoint.lat) && Number.isFinite(targetPoint.lon) && map) {
        const centerLng = map.getCenter().lng;
        const wrappedLon = targetPoint.lon + Math.round((centerLng - targetPoint.lon) / 360) * 360;
        map.flyTo([targetPoint.lat, wrappedLon], 6, {
          animate: true,
          duration: 1.2
        });
      }

      // If "All Storms" basin layer is on, refresh highlight
      if (layerVisibility.allStorms) {
        toggleAllStormsLayer(true);
      }
    } catch (err) {
      console.error(err);
      alert('Unable to load replay for this storm.');
    }
  }

  // ==========================================================================
  // 8. INTERACTIVE HUD CONTROLS & EVENT LISTENERS
  // ==========================================================================

  function setupHudControls() {
    // Left HUD Collapse Toggle
    const leftHud = document.getElementById('left-hud');
    const toggleLeftBtn = document.getElementById('btn-toggle-left-hud');
    if (toggleLeftBtn && leftHud) {
      toggleLeftBtn.onclick = () => {
        const isCollapsed = leftHud.classList.toggle('collapsed');
        document.body.classList.toggle('sidebar-collapsed', isCollapsed);
      };
    }

    // Sidebar Tab Navigation
    document.querySelectorAll('.hud-tab').forEach(tab => {
      tab.onclick = () => {
        document.querySelectorAll('.hud-tab').forEach(t => {
          t.classList.remove('active');
          t.setAttribute('aria-selected', 'false');
        });
        document.querySelectorAll('.tab-pane').forEach(p => {
          p.classList.remove('active');
          p.hidden = true;
        });

        tab.classList.add('active');
        tab.setAttribute('aria-selected', 'true');
        const targetPane = document.getElementById(`pane-${tab.dataset.tab}`);
        if (targetPane) {
          targetPane.classList.add('active');
          targetPane.hidden = false;
        }
      };
    });

    // Basemap Toggles (Satellite, Dark, Hybrid)
    document.querySelectorAll('[data-map-layer]').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('[data-map-layer]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        switchBasemap(btn.dataset.mapLayer);
      };
    });

    // Cloud Simulation Mode Toggles (Visible, Infrared, Radar)
    document.querySelectorAll('[data-cloud-mode]').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('[data-cloud-mode]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        cloudMode = btn.dataset.cloudMode;
      };
    });

    // Layer Visibility Toggles (Right HUD)
    document.querySelectorAll('.layer-toggle-btn').forEach(btn => {
      btn.onclick = async () => {
        const layer = btn.dataset.layer;
        layerVisibility[layer] = !layerVisibility[layer];
        btn.classList.toggle('active', layerVisibility[layer]);

        if (layer === 'track' && trackLayerGroup) {
          if (layerVisibility.track) {
            renderVectorLayers(currentStorm);
          } else {
            trackLayerGroup.clearLayers();
            forecastLayerGroup.clearLayers();
          }
        } else if (layer === 'cone' && coneLayerGroup) {
          if (layerVisibility.cone) {
            renderVectorLayers(currentStorm);
          } else {
            coneLayerGroup.clearLayers();
          }
        } else if (layer === 'radii' && radiiLayerGroup) {
          if (layerVisibility.radii) {
            renderVectorLayers(currentStorm);
          } else {
            radiiLayerGroup.clearLayers();
          }
        } else if (layer === 'borders' && bordersTileLayer) {
          if (layerVisibility.borders) {
            bordersTileLayer.addTo(map);
          } else {
            map.removeLayer(bordersTileLayer);
          }
        } else if (layer === 'allStorms') {
          if (layerVisibility.allStorms) {
            await loadBasinAllStorms();
            toggleAllStormsLayer(true);
          } else {
            toggleAllStormsLayer(false);
          }
        }
      };
    });

    // Map Zoom & Recenter Controls
    document.getElementById('btn-map-zoom-in')?.addEventListener('click', () => map?.zoomIn());
    document.getElementById('btn-map-zoom-out')?.addEventListener('click', () => map?.zoomOut());

    const recenterAction = () => {
      const state = getInterpolatedState();
      if (state && map) {
        const centerLng = map.getCenter().lng;
        const wrappedLon = state.lon + Math.round((centerLng - state.lon) / 360) * 360;
        map.flyTo([state.lat, wrappedLon], Math.max(6, map.getZoom()), { animate: true, duration: 0.8 });
      }
    };
    document.getElementById('btn-map-recenter')?.addEventListener('click', recenterAction);
    document.getElementById('btn-center-cyclone')?.addEventListener('click', recenterAction);

    document.getElementById('btn-map-reset-basin')?.addEventListener('click', () => {
      map?.flyTo([14.5, 84.5], 5, { animate: true, duration: 1.0 });
    });

    // IMD Legend Toggle
    const legendBtn = document.getElementById('btn-legend-toggle');
    const legendContent = document.getElementById('legend-content');
    if (legendBtn && legendContent) {
      legendBtn.onclick = () => {
        const isHidden = legendContent.hidden;
        legendContent.hidden = !isHidden;
        legendBtn.setAttribute('aria-expanded', String(!isHidden));
        legendBtn.querySelector('.legend-chevron').textContent = isHidden ? '▾' : '▸';
      };
    }

    // Playback Controls (Play, Prev, Next, Scrubber, Speed)
    document.getElementById('btn-sim-play')?.addEventListener('click', togglePlaySimulation);

    document.getElementById('btn-sim-prev')?.addEventListener('click', () => {
      pauseSimulation();
      simIndex = Math.max(0, Math.floor(simIndex) - 1);
      updateSimulationUI();
    });

    document.getElementById('btn-sim-next')?.addEventListener('click', () => {
      pauseSimulation();
      simIndex = Math.min(timelineSequence.length - 1, Math.floor(simIndex) + 1);
      updateSimulationUI();
    });

    document.getElementById('btn-sim-start')?.addEventListener('click', () => {
      pauseSimulation();
      simIndex = 0;
      updateSimulationUI();
    });

    document.getElementById('btn-sim-end')?.addEventListener('click', () => {
      pauseSimulation();
      simIndex = timelineSequence.length - 1;
      updateSimulationUI();
    });

    // Scrubber Slider
    const slider = document.getElementById('timeline-slider');
    if (slider) {
      slider.addEventListener('input', e => {
        pauseSimulation();
        simIndex = Number(e.target.value);
        updateSimulationUI();
      });
    }

    // Speed Selector Buttons
    document.querySelectorAll('.speed-btn').forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        playbackSpeed = Number(btn.dataset.speed) || 1.0;
      };
    });

    // Loop Checkbox
    const loopChk = document.getElementById('chk-sim-loop');
    if (loopChk) {
      loopChk.addEventListener('change', () => {
        isLooping = loopChk.checked;
      });
    }

    // Research Brief Modal
    setupBriefModal();

    // Fullscreen Toggle
    const fsBtn = document.getElementById('btn-fullscreen');
    if (fsBtn) {
      fsBtn.onclick = () => {
        if (!document.fullscreenElement) {
          document.documentElement.requestFullscreen().catch(() => {});
        } else {
          document.exitFullscreen().catch(() => {});
        }
      };
    }
  }

  function setupBriefModal() {
    const modal = document.getElementById('bulletin-modal');
    const openBtn = document.getElementById('btn-bulletin');
    const closeBtn = document.getElementById('btn-close-modal');
    const copyBtn = document.getElementById('btn-copy-bulletin');
    const downloadBtn = document.getElementById('btn-download-bulletin');
    const content = document.getElementById('bulletin-content');

    if (!modal) return;

    const closeModal = () => { modal.hidden = true; };

    if (openBtn) {
      openBtn.onclick = () => {
        if (content) content.textContent = currentStorm?.bulletin_text || 'Research brief is currently unavailable.';
        modal.hidden = false;
      };
    }

    if (closeBtn) closeBtn.onclick = closeModal;
    modal.onclick = e => { if (e.target === modal) closeModal(); };

    if (copyBtn) {
      copyBtn.onclick = async () => {
        if (currentStorm?.bulletin_text) {
          await navigator.clipboard.writeText(currentStorm.bulletin_text);
          copyBtn.textContent = '✓ Copied!';
          setTimeout(() => { copyBtn.textContent = '📋 Copy Text'; }, 1800);
        }
      };
    }

    if (downloadBtn) {
      downloadBtn.onclick = () => {
        const text = currentStorm?.bulletin_text || '';
        const blob = new Blob([text], { type: 'text/plain' });
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `cyclone-research-brief-${currentStorm?.id || 'storm'}.txt`;
        link.click();
        URL.revokeObjectURL(link.href);
      };
    }
  }

  // ==========================================================================
  // 9. APPLICATION ENTRY POINT
  // ==========================================================================

  document.addEventListener('DOMContentLoaded', async () => {
    initMap();
    setupHudControls();
    setupStormSelector();
    renderSidebarSources();

    try {
      await loadStormList();
      await loadStorm('current');
    } catch (err) {
      console.error('Initialization error:', err);
    }
  });

})();
