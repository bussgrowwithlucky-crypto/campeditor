const API_BASE = window.API_BASE || "";

const form = document.getElementById("job-form");
const renderBtn = document.getElementById("render-btn");
const statusPanel = document.getElementById("status-panel");
const progressFill = document.getElementById("progress-fill");
const statusText = document.getElementById("status-text");
const etaText = document.getElementById("eta-text");
const errorText = document.getElementById("error-text");
const downloadLink = document.getElementById("download-link");
const preview = document.getElementById("preview");
const variationsPanel = document.getElementById("variations");
const brollPackPanel = document.getElementById("broll-pack-panel");
const brollPackHeading = document.getElementById("broll-pack-heading");
const manualTitleInput = document.getElementById("manual-title");
const trimRow = document.getElementById("trim-row");
const startInput = document.getElementById("start");
const endInput = document.getElementById("end");
const replicateCheckbox = document.getElementById("replicate");
const replicateFields = document.getElementById("replicate-fields");
const referenceInput = document.getElementById("reference");
const referenceUrlInput = document.getElementById("reference-url");
const musicInput = document.getElementById("music");
const brollPackCheckbox = document.getElementById("broll-pack");
const enableLearnedBrollCheckbox = document.getElementById("enable-learned-broll");
const useIntelligentSelectorCheckbox = document.getElementById("use-intelligent-selector");
const addCaptionCheckbox = document.getElementById("add-caption");
const brollSourceFrameio = document.getElementById("broll-source-frameio");
const brollSourceYoutube = document.getElementById("broll-source-youtube");
const brollSourceBoth = document.getElementById("broll-source-both");
const brollFrameio2Row = document.getElementById("broll-frameio-2-row");
const brollFrameio2Checkbox = document.getElementById("broll-frameio-2");

replicateCheckbox.addEventListener("change", () => {
  replicateFields.hidden = !replicateCheckbox.checked;
});

// B-roll source checkboxes: "Both" mirrors the two individual boxes.
// Checking Both checks Frame.io + YouTube; unchecking either individual
// box unchecks Both; checking both individual boxes re-checks Both.
brollSourceBoth.addEventListener("change", () => {
  brollSourceFrameio.checked = brollSourceBoth.checked;
  brollSourceYoutube.checked = brollSourceBoth.checked;
  updateBrollFrameio2Visibility();
});
[brollSourceFrameio, brollSourceYoutube].forEach((box) => {
  box.addEventListener("change", () => {
    brollSourceBoth.checked = brollSourceFrameio.checked && brollSourceYoutube.checked;
    updateBrollFrameio2Visibility();
  });
});

// The "Also search folder 2" checkbox only makes sense when the Frame.io
// library rung is active (Frame.io or Both); hide it outright for the
// YouTube-only case so the UI can't advertise a no-op. The unchecked state
// is the default; toggling Frame.io back on restores the user's last choice.
function updateBrollFrameio2Visibility() {
  if (!brollFrameio2Row) return;
  brollFrameio2Row.hidden = !(brollSourceFrameio.checked || brollSourceBoth.checked);
}
updateBrollFrameio2Visibility();

function resolveBrollSource() {
  // null = nothing selected (validation error in the submit handler).
  if (brollSourceFrameio.checked && brollSourceYoutube.checked) return "both";
  if (brollSourceFrameio.checked) return "frameio";
  if (brollSourceYoutube.checked) return "youtube";
  return null;
}

let pollTimer = null;

document.querySelectorAll('input[name="clip_mode"]').forEach((radio) => {
  radio.addEventListener("change", updateClipMode);
});

document.querySelectorAll('input[name="title_mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    manualTitleInput.hidden = form.title_mode.value !== "manual";
  });
});

updateClipMode();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearInterval(pollTimer);

  const file = document.getElementById("file").files[0];
  if (!file) {
    showError("Choose a video file first — the main landscape clip to turn into a short.");
    return;
  }

  const data = new FormData();
  data.append("file", file);
  data.append("clip_mode", form.clip_mode.value);
  data.append("start", startInput.value);
  data.append("end", endInput.value);
  data.append("title_mode", form.title_mode.value);
  data.append("manual_title", manualTitleInput.value);
  data.append("color_grade", document.getElementById("color-grade").value);
  data.append("replicate", replicateCheckbox.checked ? "true" : "false");
  // Per-job opt-out for the auto-learned-B-roll insertion (non-replicate
  // jobs only). Defaults to True to preserve the existing flow; users who
  // want a plain caption + title render uncheck it before submitting.
  data.append(
    "enable_learned_broll",
    enableLearnedBrollCheckbox.checked ? "true" : "false",
  );
  if (replicateCheckbox.checked) {
    if (!referenceUrlInput.value.trim() && !referenceInput.files[0]) {
      showError("Replicate needs a reference link or file");
      return;
    }
    const brollSource = resolveBrollSource();
    if (!brollSource) {
      showError("Pick at least one B-roll source (Frame.io, YouTube, or Both)");
      return;
    }
    data.append("broll_source", brollSource);
    // Per-replicate-job opt-in to merge the secondary Frame.io share
    // (BROLL_FRAMEIO_SHARE_URL_2) into this job's B-roll library. Ignored by
    // the backend when YouTube-only is selected or when no secondary URL is
    // configured, so the UI doesn't have to gate this on either condition.
    data.append(
      "use_broll_frameio_2",
      brollFrameio2Checkbox.checked ? "true" : "false",
    );
    data.append("reference_url", referenceUrlInput.value.trim());
    if (referenceInput.files[0]) {
      data.append("reference", referenceInput.files[0]);
    }
    if (musicInput.files[0]) {
      data.append("music", musicInput.files[0]);
    }
    // broll_pack is only meaningful when the job is in replicate mode (the
    // pack is sourced from the reference's detected cutaways). The backend
    // ignores the flag otherwise, but we send it only when replicate is on
    // so the field stays out of the non-replicate request entirely.
    data.append("broll_pack", brollPackCheckbox.checked ? "true" : "false");
    // Intelligent selector only kicks in inside the B-roll matching code
    // path (which is itself only reached in replicate mode). Sending the
    // flag on every replicate job means the user can leave the box checked
    // and forget about it.
    data.append(
      "use_intelligent_selector",
      useIntelligentSelectorCheckbox.checked ? "true" : "false",
    );
    // Per-replicate-job caption toggle. Defaults to True so historical
    // behavior is preserved; unchecked users get a title-only render.
    data.append(
      "add_caption",
      addCaptionCheckbox.checked ? "true" : "false",
    );
  }

  resetStatus("Uploading...", 0.05);
  renderBtn.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/api/jobs/upload`, { method: "POST", body: data });
    const payload = await readResponse(response);
    pollFailures = 0;
    pollTimer = setInterval(() => poll(payload.id), 1500);
  } catch (err) {
    showError(err.message);
  }
});

function updateClipMode() {
  const manual = form.clip_mode.value === "manual";
  trimRow.hidden = !manual;
  startInput.disabled = !manual;
  endInput.disabled = !manual;
  startInput.required = manual;
  endInput.required = manual;
}

// Consecutive failed polls. A single "Failed to fetch" (e.g. the dev server
// reloading after a code change) must NOT kill the poll loop permanently —
// the job keeps running server-side. Only give up after several in a row.
let pollFailures = 0;
const MAX_POLL_FAILURES = 8;

async function poll(jobId) {
  try {
    const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
    const job = await readResponse(response);
    pollFailures = 0;
    progressFill.style.width = `${Math.round(job.progress * 100)}%`;
    statusText.textContent = `${job.status.toUpperCase()} - ${job.message}`;
    updateWarning(job.warning);
    updateEta(job);

    if (job.status === "ready") {
      clearInterval(pollTimer);
      renderBtn.disabled = false;
      etaText.hidden = true;
      const urls = job.variation_urls && job.variation_urls.length
        ? job.variation_urls
        : (job.output_url ? [job.output_url] : []);
      if (urls.length > 1) {
        renderVariations(urls);
      } else if (urls.length === 1) {
        downloadLink.href = API_BASE + urls[0];
        downloadLink.hidden = false;
        preview.src = API_BASE + urls[0];
        preview.hidden = false;
      }
      // B-roll pack panel: only present when the job ran in broll_pack mode
      // AND the pipeline emitted at least one trimmed clip. Rendered AFTER
      // the variation grid so the main video preview is what the user sees
      // first — pack is the secondary deliverable.
      const packUrls = job.broll_pack_urls || [];
      if (packUrls.length) {
        renderBrollPack(packUrls);
      }
    } else if (job.status === "failed") {
      clearInterval(pollTimer);
      etaText.hidden = true;
      showError(job.error || job.message || "Job failed");
    }
  } catch (err) {
    pollFailures += 1;
    if (pollFailures >= MAX_POLL_FAILURES) {
      clearInterval(pollTimer);
      showError(`Lost contact with the server (${err.message}). The job may still be running — refresh to check.`);
    }
    // Transient failure (server reload / brief network blip): keep polling.
  }
}

function updateWarning(warning) {
  // Non-fatal notice (e.g. "YouTube source unavailable — used Frame.io
  // only"). Lives in its own element so it survives status-line updates.
  let el = document.getElementById("warning-text");
  if (!warning) {
    if (el) el.hidden = true;
    return;
  }
  if (!el) {
    el = document.createElement("p");
    el.id = "warning-text";
    el.style.color = "#b58900";
    el.style.margin = "0.25rem 0 0";
    statusText.insertAdjacentElement("afterend", el);
  }
  el.textContent = `⚠ ${warning}`;
  el.hidden = false;
}

function updateEta(job) {
  const secs = job.eta_seconds;
  if (secs == null || !Number.isFinite(secs) || secs <= 0 || job.status === "ready" || job.status === "failed") {
    etaText.hidden = true;
    return;
  }
  const total = Math.round(secs);
  const mins = Math.floor(total / 60);
  const rem = total % 60;
  const pretty = mins > 0 ? `${mins}m ${rem}s` : `${rem}s`;
  etaText.textContent = `Estimated time remaining: ~${pretty}`;
  etaText.hidden = false;
}

function renderVariations(urls) {
  variationsPanel.innerHTML = "";
  urls.forEach((url, index) => {
    const card = document.createElement("div");
    card.className = "variation-card";

    const label = document.createElement("p");
    label.className = "variation-label";
    label.textContent = `Variation ${index + 1} of ${urls.length}`;

    const video = document.createElement("video");
    video.controls = true;
    video.src = API_BASE + url;

    const link = document.createElement("a");
    link.className = "download";
    link.href = API_BASE + url;
    link.setAttribute("download", "");
    link.textContent = "Download";

    card.append(label, video, link);
    variationsPanel.append(card);
  });
  variationsPanel.hidden = false;
}

// Render the B-Roll Pack panel. `items` is the JobSummary.broll_pack_urls
// array — each entry carries span_index/rank/start/end/query + the API
// download URL. One card per item, label = "Cutaway N · option R · m:ss–m:ss"
// so the user can drop the file straight into Premiere without re-mapping
// timestamps.
function renderBrollPack(items) {
  brollPackPanel.innerHTML = "";
  items.forEach((item, index) => {
    const card = document.createElement("div");
    card.className = "variation-card";

    const label = document.createElement("p");
    label.className = "variation-label";
    const startStr = formatTimestamp(item.start);
    const endStr = formatTimestamp(item.end);
    label.textContent = `Cutaway ${item.span_index + 1} \u00B7 option ${item.rank} \u00B7 ${startStr}\u2013${endStr}`;

    const query = document.createElement("p");
    query.className = "pack-query";
    query.textContent = item.query || "";

    const video = document.createElement("video");
    video.controls = true;
    video.src = API_BASE + item.url;

    const link = document.createElement("a");
    link.className = "download";
    link.href = API_BASE + item.url;
    link.setAttribute("download", "");
    link.textContent = "Download";

    card.append(label, query, video, link);
    brollPackPanel.append(card);
  });
  brollPackPanel.hidden = false;
  brollPackHeading.hidden = false;
}

// 0:04-style timestamp for pack labels — keeps the on-screen trim range
// scannable without hauling in a 5kB date library for one line of UI text.
function formatTimestamp(seconds) {
  const total = Math.max(0, Math.round(Number(seconds) || 0));
  const mins = Math.floor(total / 60);
  const secs = total % 60;
  return `${mins}:${secs.toString().padStart(2, "0")}`;
}

async function readResponse(response) {
  const text = await response.text();
  let payload = {};
  try { payload = JSON.parse(text); } catch { /* non-JSON error body */ }
  if (!response.ok) throw new Error(payload.detail || text || `HTTP ${response.status}`);
  return payload;
}

function resetStatus(message, progress) {
  statusPanel.hidden = false;
  errorText.hidden = true;
  etaText.hidden = true;
  downloadLink.hidden = true;
  preview.hidden = true;
  preview.removeAttribute("src");
  variationsPanel.hidden = true;
  variationsPanel.innerHTML = "";
  brollPackPanel.hidden = true;
  brollPackPanel.innerHTML = "";
  brollPackHeading.hidden = true;
  statusText.textContent = message;
  progressFill.style.width = `${Math.round(progress * 100)}%`;
  statusPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}

function showError(message) {
  renderBtn.disabled = false;
  statusPanel.hidden = false;
  errorText.textContent = message;
  errorText.hidden = false;
  statusPanel.scrollIntoView({ behavior: "smooth", block: "center" });
}
