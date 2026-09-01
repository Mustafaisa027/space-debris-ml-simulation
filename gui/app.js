"use strict";

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
  data: null,
  scenarioIndex: 0,
  minute: 0,
  playing: false,
  speed: 1,
  view: "orbit",
  metric: "pr_auc",
  lastFrame: 0,
  tcaPulseUntil: 0,
  tcaPauseUsed: false,
  reducedMotion: matchMedia("(prefers-reduced-motion: reduce)").matches,
  stars: [],
};

const elements = {
  shell: $("#appShell"),
  loading: $("#loadingScreen"),
  scenario: $("#scenarioSelect"),
  orbitCanvas: $("#orbitCanvas"),
  distanceCanvas: $("#distanceChart"),
  distanceReadout: $("#distanceReadout"),
  emptyChart: $("#emptyChart"),
  play: $("#playButton"),
  slider: $("#timeSlider"),
  timeOutput: $("#timeOutput"),
  tcaMarker: $("#tcaMarker"),
  tcaToast: $("#tcaToast"),
  currentUtc: $("#currentUtc"),
  countdown: $("#tcaCountdown strong"),
  modelBars: $("#modelBars"),
  qualityGates: $("#qualityGates"),
  archiveCounts: $("#archiveCounts"),
};

function currentScenario() {
  return state.data.replay.scenarios[state.scenarioIndex];
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function formatNumber(value, decimals = 3) {
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

function formatDuration(minutes) {
  const absolute = Math.abs(minutes);
  if (absolute < 1) return `${Math.round(absolute * 60)} s`;
  if (absolute < 60) return `${formatNumber(absolute, 1)} min`;
  return `${formatNumber(absolute / 60, 2)} h`;
}

function formatClock(minutes) {
  const total = Math.max(0, Math.round(minutes));
  const hours = Math.floor(total / 60);
  const mins = total % 60;
  return `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}`;
}

function utcAtMinute(scenario, minute) {
  const start = Date.parse(scenario.snapshot_utc);
  if (!Number.isFinite(start)) return "—";
  return new Date(start + minute * 60000).toISOString().replace(".000Z", "Z");
}

function seededRandom(seed) {
  let value = seed >>> 0;
  return () => {
    value = (value * 1664525 + 1013904223) >>> 0;
    return value / 4294967296;
  };
}

function prepareStars() {
  const random = seededRandom(114764);
  state.stars = Array.from({ length: 110 }, () => ({
    x: random(),
    y: random() * 0.78,
    r: 0.35 + random() * 1.2,
    a: 0.16 + random() * 0.58,
  }));
}

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(rect.width * ratio));
  const height = Math.max(1, Math.floor(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function quadraticPoint(start, control, end, t) {
  const one = 1 - t;
  return {
    x: one * one * start.x + 2 * one * t * control.x + t * t * end.x,
    y: one * one * start.y + 2 * one * t * control.y + t * t * end.y,
  };
}

function drawQuadratic(ctx, start, control, end, color, width = 1) {
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  ctx.quadraticCurveTo(control.x, control.y, end.x, end.y);
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.stroke();
}

function drawEarth(ctx, width, height) {
  const radius = Math.max(width * 0.49, 310);
  const cx = width * 0.5;
  const cy = height + radius * 0.76;
  const glow = ctx.createRadialGradient(cx, cy - radius, 0, cx, cy - radius, radius * 0.22);
  glow.addColorStop(0, "rgba(85,221,242,.28)");
  glow.addColorStop(1, "rgba(85,221,242,0)");
  ctx.fillStyle = glow;
  ctx.fillRect(0, height - radius * 0.42, width, radius * 0.42);

  const earth = ctx.createRadialGradient(cx - radius * 0.25, cy - radius * 0.33, radius * 0.05, cx, cy, radius);
  earth.addColorStop(0, "#173d5a");
  earth.addColorStop(0.45, "#0b263d");
  earth.addColorStop(1, "#030812");
  ctx.beginPath();
  ctx.arc(cx, cy, radius, 0, Math.PI * 2);
  ctx.fillStyle = earth;
  ctx.fill();
  ctx.strokeStyle = "rgba(85,221,242,.72)";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.save();
  ctx.clip();
  ctx.strokeStyle = "rgba(85,221,242,.09)";
  ctx.lineWidth = 0.8;
  for (let i = -3; i <= 3; i += 1) {
    ctx.beginPath();
    ctx.ellipse(cx, cy + i * radius * 0.11, radius, radius * (0.13 + Math.abs(i) * 0.015), 0, 0, Math.PI * 2);
    ctx.stroke();
  }
  ctx.restore();
}

function drawObject(ctx, point, type, label, catalogId, color) {
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = 16;
  ctx.fillStyle = color;
  if (type === "diamond") {
    ctx.translate(point.x, point.y);
    ctx.rotate(Math.PI / 4);
    ctx.fillRect(-5, -5, 10, 10);
    ctx.rotate(-Math.PI / 4);
    ctx.translate(-point.x, -point.y);
  } else {
    ctx.beginPath();
    ctx.arc(point.x, point.y, 5.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
  ctx.font = "600 11px system-ui, sans-serif";
  ctx.fillStyle = "#eef6ff";
  ctx.fillText(label, point.x + 11, point.y - 7);
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#93a8c2";
  ctx.fillText(catalogId ? `NORAD ${catalogId}` : "catalogued object", point.x + 11, point.y + 7);
}

function drawOrbitView(ctx, width, height, scenario) {
  const background = ctx.createLinearGradient(0, 0, 0, height);
  background.addColorStop(0, "#030711");
  background.addColorStop(1, "#071421");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  for (const star of state.stars) {
    ctx.globalAlpha = star.a;
    ctx.fillStyle = "#dff8ff";
    ctx.fillRect(star.x * width, star.y * height, star.r, star.r);
  }
  ctx.globalAlpha = 1;
  drawEarth(ctx, width, height);

  const event = { x: width * 0.52, y: height * 0.43 };
  const pathA = {
    start: { x: -30, y: height * 0.64 },
    control: { x: width * 0.37, y: height * 0.04 },
    end: { x: width + 35, y: height * 0.45 },
  };
  const pathB = {
    start: { x: width + 30, y: height * 0.18 },
    control: { x: width * 0.54, y: height * 0.78 },
    end: { x: -35, y: height * 0.29 },
  };
  drawQuadratic(ctx, pathA.start, pathA.control, pathA.end, "rgba(85,221,242,.42)", 1.15);
  drawQuadratic(ctx, pathB.start, pathB.control, pathB.end, "rgba(255,177,74,.38)", 1.15);
  drawQuadratic(ctx, { x: -20, y: height * 0.37 }, { x: width * 0.46, y: height * 0.02 }, { x: width + 20, y: height * 0.3 }, "rgba(147,168,194,.15)", 0.8);

  const horizon = Number(elements.slider.max) || 720;
  const tca = scenario.time_to_tca_min;
  const relative = (state.minute - tca) / Math.max(horizon * 0.72, 1);
  const t = clamp(0.5 + relative, 0.03, 0.97);
  const pointA = quadraticPoint(pathA.start, pathA.control, pathA.end, t);
  const pointB = quadraticPoint(pathB.start, pathB.control, pathB.end, t);
  const nearTca = Math.abs(state.minute - tca) < Math.max(3, horizon * 0.008);
  if (nearTca) {
    pointA.x = event.x - 7;
    pointA.y = event.y - 3;
    pointB.x = event.x + 7;
    pointB.y = event.y + 3;
  }

  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.strokeStyle = nearTca ? "rgba(255,177,74,.95)" : "rgba(147,168,194,.23)";
  ctx.lineWidth = nearTca ? 1.3 : 0.8;
  ctx.beginPath();
  ctx.moveTo(pointA.x, pointA.y);
  ctx.lineTo(pointB.x, pointB.y);
  ctx.stroke();
  ctx.restore();

  if (nearTca || performance.now() < state.tcaPulseUntil) {
    const pulse = state.reducedMotion ? 0.5 : (Math.sin(performance.now() / 115) + 1) / 2;
    ctx.beginPath();
    ctx.arc(event.x, event.y, 18 + pulse * 13, 0, Math.PI * 2);
    ctx.strokeStyle = `rgba(255,177,74,${0.32 + pulse * 0.28})`;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }

  drawObject(ctx, pointA, "circle", scenario.object_1, scenario.catalog_id_1, "#55ddf2");
  drawObject(ctx, pointB, "diamond", scenario.object_2, scenario.catalog_id_2, "#ffb14a");

  // Anchored to a fixed corner rather than the converging point: near TCA the
  // objects and their labels sit right where an event-relative readout would
  // be drawn, which made this text overlap the object labels at the exact
  // moment the audience is looking most closely.
  ctx.fillStyle = "rgba(147,168,194,.85)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`TCA ${scenario.tca_utc || "—"}`, 18, height - 34);
  ctx.fillStyle = "#ffb14a";
  ctx.fillText(`MISS ${formatNumber(scenario.min_distance_km, 3)} km`, 18, height - 18);
}

function drawEncounterView(ctx, width, height, scenario) {
  const background = ctx.createRadialGradient(width * 0.5, height * 0.46, 0, width * 0.5, height * 0.46, width * 0.75);
  background.addColorStop(0, "#102237");
  background.addColorStop(1, "#030711");
  ctx.fillStyle = background;
  ctx.fillRect(0, 0, width, height);

  const center = { x: width * 0.5, y: height * 0.48 };
  const axisLength = Math.min(width, height) * 0.32;
  const axes = [
    { dx: 1, dy: 0, label: "I · IN-TRACK" },
    { dx: 0, dy: -1, label: "R · RADIAL" },
    { dx: -0.63, dy: 0.52, label: "C · CROSS-TRACK" },
  ];
  ctx.font = "10px ui-monospace, monospace";
  axes.forEach((axis) => {
    ctx.beginPath();
    ctx.moveTo(center.x, center.y);
    ctx.lineTo(center.x + axis.dx * axisLength, center.y + axis.dy * axisLength);
    ctx.strokeStyle = "rgba(147,168,194,.32)";
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = "#93a8c2";
    ctx.fillText(axis.label, center.x + axis.dx * axisLength + 7, center.y + axis.dy * axisLength - 5);
  });

  const vector = {
    x: scenario.relative_intrack_km,
    y: -scenario.relative_radial_km,
  };
  const magnitude = Math.max(Math.hypot(vector.x, vector.y), 0.001);
  const scale = axisLength * 0.7 / magnitude;
  const target = { x: center.x + vector.x * scale, y: center.y + vector.y * scale };
  ctx.save();
  ctx.setLineDash([6, 5]);
  ctx.strokeStyle = "rgba(255,177,74,.9)";
  ctx.beginPath();
  ctx.moveTo(center.x, center.y);
  ctx.lineTo(target.x, target.y);
  ctx.stroke();
  ctx.restore();

  drawObject(ctx, center, "circle", scenario.object_1, scenario.catalog_id_1, "#55ddf2");
  drawObject(ctx, target, "diamond", scenario.object_2, scenario.catalog_id_2, "#ffb14a");

  // Anchored to the bottom-left instead of the top-left: the top-left corner
  // is where the "Propagated frame" HTML overlay sits, and drawing text
  // there too made this readout collide with it and go unreadable.
  ctx.fillStyle = "#eef6ff";
  ctx.font = "600 12px system-ui, sans-serif";
  ctx.fillText("Relative position at refined TCA", 18, height - 84);
  ctx.fillStyle = "#93a8c2";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText(`R ${formatNumber(scenario.relative_radial_km, 3)} km`, 18, height - 63);
  ctx.fillText(`I ${formatNumber(scenario.relative_intrack_km, 3)} km`, 18, height - 48);
  ctx.fillText(`C ${formatNumber(scenario.relative_crosstrack_km, 3)} km`, 18, height - 33);
  ctx.fillStyle = "#ffb14a";
  ctx.fillText(`MISS ${formatNumber(scenario.min_distance_km, 3)} km`, 18, height - 18);
}

function drawOrbitCanvas() {
  if (!state.data) return;
  const scenario = currentScenario();
  const { ctx, width, height } = setupCanvas(elements.orbitCanvas);
  ctx.clearRect(0, 0, width, height);
  if (state.view === "encounter") drawEncounterView(ctx, width, height, scenario);
  else drawOrbitView(ctx, width, height, scenario);
  elements.orbitCanvas.setAttribute(
    "aria-label",
    `${scenario.replay_label}: ${scenario.object_1} and ${scenario.object_2}; minimum propagated separation ${formatNumber(scenario.min_distance_km, 3)} kilometres. Schematic, not to scale.`
  );
}

function seriesPointAtMinute(series, minute) {
  if (!series.length) return null;
  let nearest = series[0];
  for (const point of series) {
    if (Math.abs(point.minute - minute) < Math.abs(nearest.minute - minute)) nearest = point;
  }
  return nearest;
}

function drawDistanceChart() {
  if (!state.data) return;
  const scenario = currentScenario();
  const series = scenario.distance_series || [];
  const { ctx, width, height } = setupCanvas(elements.distanceCanvas);
  ctx.clearRect(0, 0, width, height);
  if (!series.length) {
    elements.emptyChart.hidden = false;
    elements.distanceReadout.textContent = "TCA geometry only";
    return;
  }
  elements.emptyChart.hidden = true;
  const margin = { left: 54, right: 16, top: 15, bottom: 28 };
  const plotW = Math.max(10, width - margin.left - margin.right);
  const plotH = Math.max(10, height - margin.top - margin.bottom);
  const maxMinute = Math.max(...series.map((point) => point.minute), 1);
  const distances = series.map((point) => Math.max(point.distance_km, 0.1));
  const minLog = Math.log10(Math.min(...distances) * 0.72);
  const maxLog = Math.log10(Math.max(...distances) * 1.15);
  const x = (minute) => margin.left + (minute / maxMinute) * plotW;
  const y = (distance) => margin.top + (1 - (Math.log10(Math.max(distance, 0.1)) - minLog) / (maxLog - minLog || 1)) * plotH;

  ctx.strokeStyle = "rgba(83,120,158,.22)";
  ctx.lineWidth = 1;
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillStyle = "#93a8c2";
  const yTicks = [25, 100, 1000, 10000].filter((value) => Math.log10(value) >= minLog && Math.log10(value) <= maxLog);
  for (const tick of yTicks) {
    const py = y(tick);
    ctx.beginPath(); ctx.moveTo(margin.left, py); ctx.lineTo(width - margin.right, py); ctx.stroke();
    ctx.fillText(tick >= 1000 ? `${tick / 1000}k` : String(tick), 8, py + 3);
  }
  [0, maxMinute / 2, maxMinute].forEach((tick) => {
    const px = x(tick);
    ctx.beginPath(); ctx.moveTo(px, margin.top); ctx.lineTo(px, height - margin.bottom); ctx.stroke();
    ctx.fillText(`${Math.round(tick / 60)}h`, px - 7, height - 9);
  });

  if (25 >= Math.pow(10, minLog) && 25 <= Math.pow(10, maxLog)) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(255,177,74,.75)";
    ctx.beginPath(); ctx.moveTo(margin.left, y(25)); ctx.lineTo(width - margin.right, y(25)); ctx.stroke();
    ctx.restore();
    ctx.fillStyle = "#ffb14a";
    ctx.fillText("25 km alarm", width - margin.right - 70, y(25) - 5);
  }

  ctx.beginPath();
  series.forEach((point, index) => {
    const px = x(point.minute), py = y(point.distance_km);
    if (index === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  });
  const lineGradient = ctx.createLinearGradient(margin.left, 0, width - margin.right, 0);
  lineGradient.addColorStop(0, "#5a8dee");
  lineGradient.addColorStop(0.55, "#55ddf2");
  lineGradient.addColorStop(1, "#5a8dee");
  ctx.strokeStyle = lineGradient;
  ctx.lineWidth = 2;
  ctx.stroke();

  const current = seriesPointAtMinute(series, state.minute);
  if (current) {
    const px = x(current.minute), py = y(current.distance_km);
    ctx.strokeStyle = "rgba(238,246,255,.32)";
    ctx.beginPath(); ctx.moveTo(px, margin.top); ctx.lineTo(px, height - margin.bottom); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fillStyle = "#55ddf2"; ctx.fill();
    elements.distanceReadout.textContent = `${formatNumber(current.distance_km, current.distance_km < 100 ? 3 : 1)} km`;
  }
}

function updateTimelineText() {
  const scenario = currentScenario();
  const horizon = Number(elements.slider.max);
  elements.timeOutput.textContent = `${formatClock(state.minute)} / ${formatClock(horizon)}`;
  elements.currentUtc.textContent = utcAtMinute(scenario, state.minute);
  const remaining = scenario.time_to_tca_min - state.minute;
  elements.countdown.textContent = remaining >= 0 ? `T− ${formatDuration(remaining)}` : `T+ ${formatDuration(remaining)}`;
  elements.tcaMarker.style.left = `${clamp((scenario.time_to_tca_min / horizon) * 100, 0, 100)}%`;
}

function updatePlaybackVisuals() {
  elements.slider.value = String(state.minute);
  updateTimelineText();
  drawOrbitCanvas();
  drawDistanceChart();
}

function setPlaying(playing) {
  state.playing = playing;
  elements.play.setAttribute("aria-pressed", String(playing));
  elements.play.innerHTML = playing
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg><span>Pause</span>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5l11 7-11 7z"/></svg><span>Play</span>';
  if (playing) requestAnimationFrame(animate);
}

function animate(now) {
  if (!state.playing) return;
  if (document.hidden) {
    state.lastFrame = now;
    requestAnimationFrame(animate);
    return;
  }
  if (!state.lastFrame) state.lastFrame = now;
  const elapsed = Math.min((now - state.lastFrame) / 1000, 0.1);
  state.lastFrame = now;
  const scenario = currentScenario();
  const previous = state.minute;
  state.minute += elapsed * 4.5 * state.speed;
  if (previous < scenario.time_to_tca_min && state.minute >= scenario.time_to_tca_min && !state.tcaPauseUsed) {
    state.minute = scenario.time_to_tca_min;
    state.tcaPulseUntil = performance.now() + 1200;
    state.tcaPauseUsed = true;
    elements.tcaToast.classList.add("show");
    setTimeout(() => elements.tcaToast.classList.remove("show"), 1000);
    if (!state.reducedMotion) {
      setPlaying(false);
      setTimeout(() => setPlaying(true), 700);
      updatePlaybackVisuals();
      return;
    }
  }
  if (state.minute >= Number(elements.slider.max)) {
    state.minute = Number(elements.slider.max);
    setPlaying(false);
  }
  updatePlaybackVisuals();
  if (state.playing) requestAnimationFrame(animate);
}

function updateEncounterPanel() {
  const scenario = currentScenario();
  $("#pairAName").textContent = scenario.object_1;
  $("#pairBName").textContent = scenario.object_2;
  $("#pairAId").textContent = scenario.catalog_id_1 ? `NORAD ${scenario.catalog_id_1}` : "Catalogued object";
  $("#pairBId").textContent = scenario.catalog_id_2 ? `NORAD ${scenario.catalog_id_2}` : "Catalogued object";
  $("#legendA").textContent = scenario.object_1;
  $("#legendB").textContent = scenario.object_2;
  $("#minDistance").textContent = `${formatNumber(scenario.min_distance_km, 3)} km`;
  $("#timeToTca").textContent = formatDuration(scenario.time_to_tca_min);
  $("#relativeSpeed").textContent = `${formatNumber(scenario.relative_velocity_km_s, 3)} km/s`;
  $("#altitudeDifference").textContent = `${formatNumber(scenario.altitude_difference_km, 3)} km`;
  $("#approachAngle").textContent = `${formatNumber(scenario.approach_angle_deg, 1)}°`;
  $("#fixedAlarm").textContent = scenario.fixed_threshold_alarm ? "TRIGGERED" : "NOT TRIGGERED";
  $("#proxyLabel").textContent = scenario.proxy_positive ? "PROXY-POSITIVE" : "PROXY-NEGATIVE";
  $("#fixedAlarm").classList.toggle("positive", scenario.fixed_threshold_alarm);
  $("#proxyLabel").classList.toggle("positive", scenario.proxy_positive);
  $("#replayLabel").textContent = scenario.replay_label;
  $("#snapshotCode").textContent = scenario.snapshot_utc;
  $("#stageTitle").textContent = `${scenario.object_1} × ${scenario.object_2}`;
  $("#sourceFooter").textContent = `Replay source: ${state.data.replay.source}`;
}

function renderModels() {
  const models = state.data.paper.models;
  const best = Math.max(...models.map((model) => model[state.metric]));
  elements.modelBars.innerHTML = models.map((model) => {
    const value = model[state.metric];
    const classes = ["model-row", model.key === "distance" ? "distance" : "", value === best ? "best" : ""].filter(Boolean).join(" ");
    return `<div class="${classes}">
      <span class="label" title="${model.label}">${model.label}</span>
      <span class="bar-track"><i class="bar-fill" style="width:${value * 100}%"></i></span>
      <span class="value">${value.toFixed(3).replace(/^0/, "")}</span>
    </div>`;
  }).join("");
  const notes = {
    pr_auc: "Distance uses continuous −minimum-distance ranking for PR-AUC.",
    f1: "Distance uses the fixed 25 km alarm for precision, recall and F1.",
    recall: "SVM maximised learned-model recall, with a substantially higher false-alarm rate.",
  };
  $("#resultsNote").textContent = notes[state.metric];
}

function renderQuality() {
  const gates = state.data.paper.quality_gates;
  elements.qualityGates.innerHTML = gates.map((gate) => {
    const progress = gate.key === "coverage" ? gate.value : gate.key === "gap" ? Math.min((gate.value / 20) * 100, 100) : Math.min((gate.value / 10) * 100, 100);
    return `<article class="gate-card">
      <div class="gate-head"><span>${gate.label}</span><b>FAILED</b></div>
      <div class="gate-value"><strong>${formatNumber(gate.value, gate.key === "identical_run" ? 0 : 3)}</strong><span>${gate.unit} · required ${gate.requirement}</span></div>
      <div class="gate-track" aria-label="${gate.label}: ${gate.value} ${gate.unit}, required ${gate.requirement}"><i style="width:${progress}%"></i></div>
    </article>`;
  }).join("");
  // Grouped the way Table 1 in the paper is: row-level counts first, then
  // pair-level counts — flat and unlabeled, the two kinds of number read as
  // one undifferentiated block instead of mirroring the paper's structure.
  const groups = [
    {
      title: "Rows",
      items: [
        ["Verified snapshots", "verified_snapshots"],
        ["Candidate observations", "candidate_observations"],
        ["Proxy-positive rows", "proxy_positive_observations"],
        ["Held-out observations", "held_out_observations"],
        ["Held-out positive rows", "held_out_positive_observations"],
      ],
    },
    {
      title: "Catalog pairs",
      items: [
        ["Active pairs", "active_catalog_pairs"],
        ["Held-out pairs", "held_out_catalog_pairs"],
      ],
    },
  ];
  elements.archiveCounts.innerHTML = groups.map((group) => `
    <div class="archive-group">
      <span class="archive-group-title">${group.title}</span>
      <dl class="archive-counts">
        ${group.items.map(([label, key]) => `<div><dt>${label}</dt><dd>${Number(state.data.paper.counts[key]).toLocaleString("en-US")}</dd></div>`).join("")}
      </dl>
    </div>`).join("");
}

function populateScenarios() {
  elements.scenario.innerHTML = state.data.replay.scenarios.map((scenario, index) => `<option value="${index}">${scenario.label}</option>`).join("");
  elements.scenario.value = String(state.scenarioIndex);
}

function applyScenario(index) {
  state.scenarioIndex = Number(index);
  state.minute = 0;
  state.tcaPauseUsed = false;
  state.tcaPulseUntil = 0;
  setPlaying(false);
  const scenario = currentScenario();
  const maxSeriesMinute = scenario.distance_series?.length ? Math.max(...scenario.distance_series.map((point) => point.minute)) : 720;
  elements.slider.max = String(Math.max(720, Math.ceil(maxSeriesMinute)));
  elements.slider.value = "0";
  updateEncounterPanel();
  updatePlaybackVisuals();
}

function selectTab(button) {
  $$(".tab-list [role=tab]").forEach((tab) => {
    const selected = tab === button;
    tab.setAttribute("aria-selected", String(selected));
    const panel = document.getElementById(tab.getAttribute("aria-controls"));
    panel.hidden = !selected;
    panel.classList.toggle("active", selected);
  });
}

function bindEvents() {
  elements.scenario.addEventListener("change", (event) => applyScenario(event.target.value));
  elements.play.addEventListener("click", () => {
    if (state.minute >= Number(elements.slider.max)) {
      state.minute = 0;
      state.tcaPauseUsed = false;
    }
    state.lastFrame = 0;
    setPlaying(!state.playing);
  });
  elements.slider.addEventListener("input", (event) => {
    state.minute = Number(event.target.value);
    state.tcaPauseUsed = state.minute >= currentScenario().time_to_tca_min;
    updatePlaybackVisuals();
  });
  $("#previousButton").addEventListener("click", () => {
    state.minute = clamp(state.minute - 5, 0, Number(elements.slider.max));
    updatePlaybackVisuals();
  });
  $("#nextButton").addEventListener("click", () => {
    state.minute = clamp(state.minute + 5, 0, Number(elements.slider.max));
    updatePlaybackVisuals();
  });
  $$("[data-speed]").forEach((button) => button.addEventListener("click", () => {
    state.speed = Number(button.dataset.speed);
    $$("[data-speed]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
  }));
  $$("[data-view]").forEach((button) => button.addEventListener("click", () => {
    state.view = button.dataset.view;
    $$("[data-view]").forEach((candidate) => {
      const selected = candidate === button;
      candidate.classList.toggle("active", selected);
      candidate.setAttribute("aria-pressed", String(selected));
    });
    drawOrbitCanvas();
  }));
  $$(".tab-list [role=tab]").forEach((button) => button.addEventListener("click", () => selectTab(button)));
  $$("[data-metric]").forEach((button) => button.addEventListener("click", () => {
    state.metric = button.dataset.metric;
    $$("[data-metric]").forEach((candidate) => candidate.classList.toggle("active", candidate === button));
    renderModels();
  }));
  $("#posterModeButton").addEventListener("click", (event) => {
    const active = !elements.shell.classList.contains("poster-mode");
    elements.shell.classList.toggle("poster-mode", active);
    event.currentTarget.setAttribute("aria-pressed", String(active));
    setTimeout(() => { drawOrbitCanvas(); drawDistanceChart(); }, 80);
  });
  $("#fullScreenButton").addEventListener("click", async () => {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen?.();
    else await document.exitFullscreen?.();
  });
  window.addEventListener("resize", () => { drawOrbitCanvas(); drawDistanceChart(); });
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "BUTTON"].includes(document.activeElement?.tagName)) return;
    if (event.code === "Space") { event.preventDefault(); elements.play.click(); }
    if (event.code === "ArrowLeft") { event.preventDefault(); $("#previousButton").click(); }
    if (event.code === "ArrowRight") { event.preventDefault(); $("#nextButton").click(); }
  });
}

async function init() {
  try {
    const response = await fetch("data/simulation.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.data = await response.json();
    prepareStars();
    populateScenarios();
    renderModels();
    renderQuality();
    bindEvents();
    applyScenario(0);
    requestAnimationFrame(() => elements.loading.classList.add("hidden"));
  } catch (error) {
    elements.loading.innerHTML = `<strong>Simulation data could not be loaded</strong><small>${String(error.message || error)} · Start the local server from gui/start_gui.ps1.</small>`;
    console.error(error);
  }
}

init();
