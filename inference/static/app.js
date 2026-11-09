// Dashboard for the local inference engine.
//
// Plain JS on a 2s poll. No framework and no bundler: the page has to render on
// a box with no network, and adding a build step to a repo whose whole point is
// "python -m inference and nothing else" would be a poor trade for a few hundred
// lines of DOM work.

const $ = (id) => document.getElementById(id);

// Stored timestamps are ISO UTC and already carry "+00:00". Appending a "Z"
// makes them invalid - which is how the session header came to read "NaNh
// NaNm" - so normalise in one place rather than at each call site.
function parseTs(iso) {
  if (!iso) return null;
  const d = new Date(/[Z+]|-\d\d:\d\d$/.test(iso) ? iso : iso + "Z");
  return isNaN(d) ? null : d;
}

const POLL_MS = 2000;

const fmt = {
  int: (n) => (n === null || n === undefined ? "—" : Math.round(n).toLocaleString()),
  one: (n) => (n === null || n === undefined ? "—" : Number(n).toFixed(1)),
  gb: (mb) => (mb === null || mb === undefined ? "—" : (mb / 1024).toFixed(1) + " GB"),
  ms: (v) => (v === null || v === undefined ? "—" : v < 1000 ? Math.round(v) + " ms" : (v / 1000).toFixed(1) + " s"),
  secs: (s) => {
    if (s === null || s === undefined) return "—";
    if (s < 60) return Math.round(s) + "s";
    if (s < 3600) return Math.floor(s / 60) + "m " + Math.round(s % 60) + "s";
    return Math.floor(s / 3600) + "h " + Math.floor((s % 3600) / 60) + "m";
  },
  pct: (v) => (v === null || v === undefined ? "—" : Math.round(v * 100) + "%"),
  // Seconds on the detail page, where turn-by-turn timing is the point.
  time: (iso) => (iso ? iso.slice(11, 19) : ""),
  // Date and minute for the overview tables. Parsed rather than sliced, so it
  // renders in the reader's timezone - the stored values are UTC, and slicing
  // the ISO string showed UTC while looking like local time.
  stamp: (iso) => {
    const d = parseTs(iso);
    if (!d) return iso || "";
    const pad = (n) => String(n).padStart(2, "0");
    return `${pad(d.getDate())}.${pad(d.getMonth() + 1)} `
         + `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  },
};

const STATE_STYLES = {
  ready:    "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/50 dark:text-emerald-300",
  starting: "bg-amber-100 text-amber-800 dark:bg-amber-900/50 dark:text-amber-300",
  idle:     "bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-300",
  stopping: "bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-300",
  failed:   "bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300",
};

const get = (path) => fetch(path).then((r) => r.json());
const post = (path, body) =>
  fetch(path, { method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body || {}) });

// A model swap kills whatever is streaming, so the button says so while a
// request is in flight. Not a confirmation dialog - the person who can be
// surprised by this is the one who forgot an agent was running elsewhere.
let inflight = 0;

function stat(label, value, extra = "") {
  return `<div>
    <div class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">${label}</div>
    <div class="mt-0.5 text-lg font-medium tabular-nums">${value}</div>
    <div class="text-xs text-slate-500 dark:text-slate-400">${extra}</div>
  </div>`;
}

function sparkline(points, width = 260, height = 32) {
  if (!points.length) return "";
  const max = Math.max(...points), min = Math.min(...points);
  const span = max - min || 1;
  const step = width / Math.max(points.length - 1, 1);
  const d = points
    .map((v, i) => `${i === 0 ? "M" : "L"}${(i * step).toFixed(1)},${(height - ((v - min) / span) * height).toFixed(1)}`)
    .join(" ");
  return `<svg viewBox="0 0 ${width} ${height}" class="w-full h-8 text-sky-500" preserveAspectRatio="none">
    <path d="${d}" fill="none" stroke="currentColor" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
  </svg>`;
}

function renderServing(state, memory, machine) {
  const pill = $("state-pill");
  pill.textContent = state.state;
  pill.className = "rounded-full px-2.5 py-0.5 text-xs font-medium " +
    (STATE_STYLES[state.state] || STATE_STYLES.idle);

  inflight = state.inflight || 0;
  $("unload").classList.toggle("hidden", state.state !== "ready");

  if (state.state === "starting") {
    $("serving").innerHTML = `
      <div class="flex items-center gap-3 mb-3">
        <span class="text-lg font-medium">${state.model || ""}</span>
        <span class="text-sm text-slate-500 dark:text-slate-400">loading…
          ${fmt.secs(state.starting_s)}</span>
      </div>
      <pre class="max-h-40 overflow-auto rounded-lg bg-slate-100 dark:bg-[#0d1117]
                  p-3 text-xs leading-relaxed text-slate-600 dark:text-slate-400
                  whitespace-pre-wrap">${(state.log_tail || "").replace(/</g, "&lt;")}</pre>`;
    return;
  }

  if (state.state !== "ready" || !state.model) {
    $("serving").innerHTML =
      `<p class="text-sm text-slate-500 dark:text-slate-400">
         Nothing loaded. Pick a model below, or just send a request — the engine
         starts it on demand.</p>`;
    return;
  }

  const rss = state.rss_mb;
  const total = (machine.ram_gb || 0) * 1024;
  const pct = rss && total ? Math.min(100, (rss / total) * 100) : 0;
  const series = memory.map((m) => m.rss_mb);

  $("serving").innerHTML = `
    <div class="grid gap-5 sm:grid-cols-2 lg:grid-cols-4 mb-5">
      ${stat("Model", state.model,
             state.adopted ? "adopted — started outside the engine" : state.engine)}
      ${stat("Uptime", fmt.secs(state.uptime_s),
             state.load_ms ? "loaded in " + fmt.ms(state.load_ms) : "")}
      ${stat("In flight", state.inflight,
             state.pid ? "pid " + state.pid : "")}
      ${stat("Memory", fmt.gb(rss), `of ${machine.ram_gb} GB unified`)}
    </div>
    <div class="h-2 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-700">
      <div class="h-full rounded-full bg-sky-500" style="width:${pct}%"></div>
    </div>
    ${sparkline(series)}`;
}

function renderModels(models, state) {
  $("models").innerHTML = models.map((m) => {
    const loaded = m._loaded;
    const label = loaded ? "Loaded"
      : inflight > 0 ? "Load (cuts current turn)" : "Load";
    return `<div class="rounded-xl border p-4 ${
      loaded ? "border-sky-400 dark:border-sky-600 bg-sky-50/50 dark:bg-sky-950/20"
             : "border-slate-200 dark:border-[#262c36] bg-white dark:bg-[#161b22]"}">
      <div class="flex items-start justify-between gap-2">
        <div class="min-w-0">
          <div class="font-medium truncate">${m.id}</div>
          <div class="text-xs text-slate-500 dark:text-slate-400">
            ${m._engine} · ${m._ctx ? (m._ctx / 1024).toFixed(0) + "K ctx" : ""}
          </div>
        </div>
        <button data-load="${m.id}" ${loaded ? "disabled" : ""}
          class="shrink-0 rounded-md px-2.5 py-1 text-xs ${
            loaded ? "text-sky-700 dark:text-sky-400 cursor-default"
                   : "border border-slate-300 dark:border-slate-600 hover:bg-slate-100 dark:hover:bg-slate-800"}"
          >${label}</button>
      </div>
      <p class="mt-2 text-xs leading-relaxed text-slate-500 dark:text-slate-400">${m._summary}</p>
    </div>`;
  }).join("");

  document.querySelectorAll("[data-load]").forEach((button) => {
    button.addEventListener("click", async () => {
      button.disabled = true;
      button.textContent = "starting…";
      await post("/api/load", { model: button.dataset.load });
      refresh();
    });
  });
}

function renderStats(rows, preempted) {
  if (!rows.length) {
    $("stats").innerHTML = `<tr><td colspan="10"
      class="px-4 py-6 text-center text-slate-500 dark:text-slate-400">
      No requests recorded yet.</td></tr>`;
    return;
  }
  $("stats").innerHTML = rows.map((r) => {
    const cut = preempted[r.alias] || 0;
    return `<tr class="border-b border-slate-100 dark:border-[#262c36] last:border-0">
      <td class="px-4 py-2.5 font-medium">${r.alias}
        <span class="text-xs text-slate-500 dark:text-slate-400">${r.engine}</span></td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.requests)}${
        cut ? `<span class="text-xs text-slate-500 dark:text-slate-400"> (${cut} cut)</span>` : ""}</td>
      <td class="px-4 py-2.5 text-right tabular-nums font-medium">${fmt.one(r.avg_tps)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums text-slate-500 dark:text-slate-400">${fmt.one(r.min_tps)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums text-slate-500 dark:text-slate-400">${fmt.one(r.max_tps)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.ms(r.avg_ttft_ms)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums text-slate-500 dark:text-slate-400">${fmt.ms(r.max_ttft_ms)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.output_tokens)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.pct(r.mtp_accept)}</td>
      <td class="px-4 py-2.5">
        <span class="rounded px-1.5 py-0.5 text-xs ${
          r.rate_source === "server"
            ? "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"
            : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
        }">${r.rate_source || "—"}</span></td>
    </tr>`;
  }).join("");
}

// Offset paging. The page number is deliberately *not* reset by the 2s refresh -
// if you have paged back to look at something, having it yanked to the top under
// you would make the pager useless. New requests do shift the offsets, so page 2
// drifts while traffic is live; that is the accepted cost of offset paging over
// keeping a cursor.
const PAGE_SIZE = 25;
let page = 0;

function renderPager(payload) {
  const total = payload.total || 0;
  const shown = payload.requests.length;
  const first = shown ? payload.offset + 1 : 0;
  $("recent-range").textContent = total
    ? `${first}–${payload.offset + shown} of ${total.toLocaleString()}`
    : "";
  $("page-prev").disabled = payload.offset === 0;
  $("page-next").disabled = payload.offset + shown >= total;
}


const CLIENT_STYLE = {
  vscode:   "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300",
  cline:    "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300",
  opencode: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
};
const clientBadge = (c) => c
  ? `<span class="rounded px-1.5 py-0.5 text-xs ${CLIENT_STYLE[c]
      || "bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400"}">${c}</span>`
  : `<span class="text-xs text-slate-400 dark:text-slate-600">&mdash;</span>`;

function renderSessions(rows) {
  if (!rows.length) {
    $("sessions").innerHTML = `<tr><td colspan="9"
      class="px-4 py-6 text-center text-slate-500 dark:text-slate-400">
      No sessions recorded yet.</td></tr>`;
    return;
  }
  $("sessions").innerHTML = rows.map((r) => {
    const notes = [];
    if (r.truncated) notes.push(`${r.truncated} truncated`);
    if (r.preempted) notes.push(`${r.preempted} cut`);
    if (r.errors) notes.push(`${r.errors} errors`);
    return `<tr data-session="${r.session_id}"
      class="cursor-pointer border-b border-slate-100 dark:border-[#262c36]
             last:border-0 hover:bg-slate-50 dark:hover:bg-slate-800/50">
      <td class="px-4 py-2.5 font-mono text-xs whitespace-nowrap">${fmt.stamp(r.started)}</td>
      <td class="px-4 py-2.5 font-medium">${r.alias}</td>
      <td class="px-4 py-2.5">${clientBadge(r.client)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.turns)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums font-medium">${fmt.one(r.tps)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.output_tokens)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.tool_calls)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums">${fmt.int(r.peak_context)}</td>
      <td class="px-4 py-2.5 text-right tabular-nums ${
        r.max_ttft_ms > 60000 ? "text-amber-600 dark:text-amber-400" : ""
      }">${fmt.ms(r.max_ttft_ms)}</td>
      <td class="px-4 py-2.5 text-xs text-amber-600 dark:text-amber-400">${notes.join(", ")}</td>
    </tr>`;
  }).join("");

  document.querySelectorAll("[data-session]").forEach((row) => {
    row.addEventListener("click", () => showSession(row.dataset.session));
  });
}

function bindTurnRows() {
  document.querySelectorAll("[data-turn]").forEach((row) => {
    row.addEventListener("click", () => showTurn(Number(row.dataset.turn)));
  });
}

function renderRecent(rows) {
  if (!rows.length) {
    $("recent").innerHTML = `<div class="px-4 py-6 text-center font-sans text-sm
      text-slate-500 dark:text-slate-400">Nothing yet.</div>`;
    return;
  }
  $("recent").innerHTML = rows.map((r) => {
    const dim = r.preempted || !r.ok;
    return `<div ${r.session_id ? `data-req-session="${r.session_id}"` : ""}
      class="flex flex-wrap items-center gap-x-4 gap-y-1 px-4 py-2 ${
        r.session_id ? "cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50" : ""} ${
        dim ? "text-slate-400 dark:text-slate-600" : ""}">
      <span class="text-slate-400 dark:text-slate-500 whitespace-nowrap">${fmt.stamp(r.ts)}</span>
      <span class="text-slate-400 dark:text-slate-500">${r.client_ip || "—"}</span>
      <span class="font-medium">${r.alias}</span>
      ${r.client ? `<span class="text-slate-400 dark:text-slate-500">${r.client}</span>` : ""}
      ${r.requested && r.requested !== r.alias
        ? `<span class="text-amber-600 dark:text-amber-400">sent ${r.requested}</span>` : ""}
      <span>ctx ${fmt.int(r.context_tokens)}</span>
      <span>out ${fmt.int(r.output_tokens)}</span>
      <span>${fmt.one(r.decode_tps)} tok/s</span>
      <span>ttft ${fmt.ms(r.ttft_ms)}</span>
      ${r.n_tool_calls ? `<span>${r.n_tool_calls} tool calls</span>` : ""}
      ${r.preempted ? `<span class="text-amber-600 dark:text-amber-400">PREEMPTED</span>` : ""}
      ${!r.ok && !r.preempted ? `<span class="text-rose-600 dark:text-rose-400">error</span>` : ""}
    </div>`;
  }).join("");

  document.querySelectorAll("[data-req-session]").forEach((row) => {
    row.addEventListener("click", () => showSession(row.dataset.reqSession));
  });
}

async function refresh() {
  if (tab !== "dashboard" || currentSession || currentTurn) return;
  try {
    const [state, models, stats, recent, memory, sessions] = await Promise.all([
      get("/api/state"), get("/api/models"), get("/api/stats"),
      get(`/api/requests?limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`),
      get("/api/memory?minutes=30"), get("/api/sessions?limit=15"),
    ]);
    $("machine").textContent =
      `${state.machine.chip} · ${state.machine.ram_gb}GB · ${state.machine.cores} cores`;
    $("endpoint").textContent = state.endpoint;
    renderServing(state, memory.series, state.machine);
    renderModels(models.data, state);
    renderStats(stats.models, stats.preempted);
    renderRecent(recent.requests);
    renderPager(recent);
    renderSessions(sessions.sessions);
    setChatEnabled(state);
  } catch (err) {
    $("state-pill").textContent = "unreachable";
    $("state-pill").className =
      "rounded-full px-2.5 py-0.5 text-xs font-medium " + STATE_STYLES.failed;
  }
}

// ---------------------------------------------------------------------------
// Quick chat: a smoke test that goes through the real path.
//
// Deliberately posts to /v1/chat/completions rather than some dashboard-only
// endpoint, and streams - so a green result here means routing, the SSE pump,
// the trace and the stats row all work, not just that the page can reach the
// server.
// ---------------------------------------------------------------------------
let chatModel = null;

function setChatEnabled(state) {
  chatModel = state.state === "ready" ? state.model : null;
  const busy = $("chat-send").dataset.busy === "1";
  $("chat-send").disabled = !chatModel || busy;
  $("chat-prompt").disabled = !chatModel;
  $("chat-hint").textContent = chatModel
    ? `Sends a streaming request to ${chatModel} through /v1/chat/completions.`
    : "Load a model first.";
}

async function sendChat(event) {
  event.preventDefault();
  const prompt = $("chat-prompt").value.trim();
  if (!prompt || !chatModel) return;

  const button = $("chat-send");
  button.dataset.busy = "1";
  button.disabled = true;
  button.textContent = "…";
  $("chat-out").classList.remove("hidden");
  $("chat-reply").textContent = "";
  $("chat-stats").textContent = "waiting for first token…";

  const started = performance.now();
  let ttft = null, tokens = 0, reasoning = 0;

  try {
    const response = await fetch("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: chatModel,
        messages: [{ role: "user", content: prompt }],
        stream: true, max_tokens: 512,
      }),
    });
    if (!response.ok) {
      const detail = await response.text();
      $("chat-reply").textContent = detail;
      $("chat-stats").textContent = `HTTP ${response.status}`;
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (payload === "[DONE]") continue;
        let event;
        try { event = JSON.parse(payload); } catch { continue; }
        const delta = (event.choices && event.choices[0] && event.choices[0].delta) || {};
        if (delta.content) {
          if (ttft === null) ttft = performance.now() - started;
          tokens += 1;
          $("chat-reply").textContent += delta.content;
        } else if (delta.reasoning_content) {
          if (ttft === null) ttft = performance.now() - started;
          reasoning += 1;
        }
      }
    }
    const wall = performance.now() - started;
    const bits = [`ttft ${fmt.ms(ttft)}`, `wall ${fmt.ms(wall)}`,
                  `${tokens} content chunks`];
    if (reasoning) bits.push(`${reasoning} reasoning chunks`);
    $("chat-stats").textContent = bits.join("  ·  ");
  } catch (err) {
    $("chat-stats").textContent = String(err);
  } finally {
    button.dataset.busy = "0";
    button.textContent = "Send";
    setChatEnabled({ state: chatModel ? "ready" : "idle", model: chatModel });
    refresh();
  }
}

$("page-prev").addEventListener("click", () => {
  page = Math.max(0, page - 1);
  refresh();
});
$("page-next").addEventListener("click", () => {
  page += 1;
  refresh();
});


// ---------------------------------------------------------------------------
// Tabs and the config panel
//
// The config tab exists for one situation: a model is stuck in a loop, you have
// killed the session, and you want to take the client's sampling out of the
// picture before trying again. So it has to work without a restart, and it has
// to still be set the next time the engine starts.
// ---------------------------------------------------------------------------
const TAB_ON = "bg-slate-200 dark:bg-slate-700 font-medium";
const TAB_OFF = "text-slate-500 dark:text-slate-400 hover:bg-slate-100 dark:hover:bg-slate-800";
let tab = "dashboard";

function showTab(name) {
  tab = name;
  document.querySelectorAll("[data-tab]").forEach((button) => {
    const on = button.dataset.tab === name;
    button.className = "tab rounded-md px-3 py-1 text-sm " + (on ? TAB_ON : TAB_OFF);
  });
  showPanel(name);
  if (name === "config") loadSettings();
  if (name === "clients") loadClients();
  if (name === "logs") loadEvents();
}

function showPanel(name) {
  // The session view is a drill-down rather than a tab, so it has no button of
  // its own - but it still has to hide the others when it opens.
  $("panel-dashboard").classList.toggle("hidden", name !== "dashboard");
  $("panel-config").classList.toggle("hidden", name !== "config");
  $("panel-clients").classList.toggle("hidden", name !== "clients");
  $("panel-session").classList.toggle("hidden", name !== "session");
  $("panel-turn").classList.toggle("hidden", name !== "turn");
  $("panel-logs").classList.toggle("hidden", name !== "logs");
}


const EVENT_STYLES = {
  tools:  "bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300",
  prompt: "bg-violet-100 text-violet-800 dark:bg-violet-900/40 dark:text-violet-300",
  strip:  "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  load:   "bg-rose-100 text-rose-800 dark:bg-rose-900/50 dark:text-rose-300",
};

async function loadEvents() {
  const { events } = await get("/api/events");
  if (!events || !events.length) {
    $("events").innerHTML = `<p class="py-3 text-sm text-slate-500 dark:text-slate-400">
      Nothing recorded yet. Events are written when something changes, so an
      engine serving steadily with settings it already reported stays quiet.</p>`;
    return;
  }
  $("events").innerHTML = events.map((e) => {
    // Drift and load failures are the two worth spotting from across the room.
    const drift = e.kind === "tools" && (e.detail?.unknown || []).length;
    const bad = drift || e.kind === "load";
    return `<div class="flex items-start gap-3 py-2.5">
      <span class="shrink-0 font-mono text-xs text-slate-400 dark:text-slate-500
                   tabular-nums">${fmt.time(e.ts)}</span>
      <span class="shrink-0 rounded px-1.5 py-0.5 text-xs font-medium ${
        EVENT_STYLES[e.kind] || EVENT_STYLES.strip}">${esc(e.kind)}</span>
      ${e.client ? `<span class="shrink-0 text-xs text-slate-500
                                 dark:text-slate-400">${esc(e.client)}</span>` : ""}
      <span class="min-w-0 grow text-sm break-words ${
        bad ? "text-amber-800 dark:text-amber-300" : "text-slate-700 dark:text-slate-300"
      }">${esc(e.text || "")}</span>
    </div>`;
  }).join("");
}


// Per-client system-prompt substitution and tool filtering. Both live on one
// card on purpose: a prompt that stops mentioning the browser tools and a set
// that drops them are the same decision, and reviewing them apart is how they
// drift. The main coding clients are always offered, then any others that
// have sent traffic or have settings - the detector knows more labels than
// you have used.
function renderPromptConfig(data) {
  const clients = data.clients || [];
  const files = data.available_prompts || [];
  const sets = data.available_toolsets || [];
  const drift = data.drift || {};
  const loops = data.loops || {};
  const contract = new Set(data.contract_blocks || []);

  if (!clients.length || !files.length) {
    $("prompt-clients").innerHTML = `<p class="text-sm text-slate-500 dark:text-slate-400">${
      !files.length ? "No prompt files in inference/prompts/."
                    : "No clients have sent traffic yet."}</p>`;
    return;
  }

  $("prompt-clients").innerHTML = clients.map((client) => {
    const cfg = (data.prompts || {})[client] || {};
    const chosen = files.find((f) => f.file === cfg.file);
    const blocks = chosen ? chosen.blocks : [];
    const active = new Set(cfg.replace || []);
    const toolset = (data.toolsets || {})[client] || "";
    const set = sets.find((s) => s.file === toolset);
    // Drift: names the set asks for that this client has stopped sending. The
    // filter cannot tell a rename from a removal, so the model has silently
    // lost the tool either way - and the guard does not fire, because the
    // other sixteen names still match.
    const missing = (drift[client] || {}).unknown || [];
    const loopAt = loops[client] || 0;
    return `<div class="rounded-xl border border-slate-200 dark:border-[#262c36]
                        bg-white dark:bg-[#161b22] p-4">
      <div class="flex flex-wrap items-center gap-3">
        ${clientBadge(client)}
        <select data-prompt-client="${client}"
                class="rounded-md border border-slate-300 dark:border-slate-600
                       bg-transparent px-2 py-1 text-sm">
          <option value="">pass through unchanged</option>
          ${files.map((f) => `<option value="${f.file}" ${
            f.file === cfg.file ? "selected" : ""}>${f.file}</option>`).join("")}
        </select>
        ${chosen ? `<span class="text-xs text-slate-500 dark:text-slate-400">${chosen.summary}</span>` : ""}
      </div>
      ${sets.length ? `<div class="mt-3 flex flex-wrap items-center gap-3">
        <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">tools</span>
        <select data-toolset-client="${client}"
                class="rounded-md border border-slate-300 dark:border-slate-600
                       bg-transparent px-2 py-1 text-sm">
          <option value="">offer every tool the client sends</option>
          ${sets.map((s) => `<option value="${s.file}" ${
            s.file === toolset ? "selected" : ""}>${s.file} (${s.count})</option>`).join("")}
        </select>
        ${set ? `<span class="text-xs text-slate-500 dark:text-slate-400">${set.summary}</span>` : ""}
      </div>
      <div class="mt-3 flex flex-wrap items-center gap-3">
        <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">loops</span>
        <label class="flex items-center gap-2 text-sm">
          collapse after
          <input type="number" min="0" max="99" step="1" value="${loopAt}"
                 data-loop-client="${client}"
                 class="w-16 rounded-md border border-slate-300 dark:border-slate-600
                        bg-transparent px-2 py-1 text-sm">
          identical tool calls
        </label>
        <span class="text-xs text-slate-500 dark:text-slate-400">${
          loopAt ? "0 turns it off for a real poll" : "off"}</span>
      </div>
      ${missing.length ? `<div class="mt-3 rounded-lg border border-amber-300
            dark:border-amber-800 bg-amber-50 dark:bg-amber-950/30 p-3">
        <p class="text-sm text-amber-800 dark:text-amber-300">
          <span class="font-medium">${missing.length} tool${
            missing.length > 1 ? "s" : ""} in this set ${
            missing.length > 1 ? "are" : "is"} no longer offered</span>
          by ${esc(client)}, so the model has lost ${
            missing.length > 1 ? "them" : "it"}. A client release renaming a
          tool looks exactly like this.
        </p>
        <p class="mt-1 font-mono text-xs text-amber-800 dark:text-amber-300">${
          missing.map(esc).join(", ")}</p>
      </div>` : ""}` : ""}
      ${blocks.length ? `<div class="mt-4 flex flex-wrap gap-3">
        ${blocks.map((b) => `<label class="flex items-center gap-2 text-sm">
          <input type="checkbox" data-prompt-block="${client}" value="${b}"
                 ${active.has(b) ? "checked" : ""}
                 class="h-4 w-4 accent-sky-600">
          <span class="font-mono text-xs">${b}</span>${
            contract.has(b) ? `<span class="text-xs text-amber-600 dark:text-amber-400"
              title="the client parses replies against this">contract</span>` : ""}
        </label>`).join("")}
      </div>` : ""}
    </div>`;
  }).join("");

  const save = (client) => {
    const file = document.querySelector(`[data-prompt-client="${client}"]`).value;
    const replace = [...document.querySelectorAll(`[data-prompt-block="${client}"]`)]
      .filter((b) => b.checked).map((b) => b.value);
    post("/api/settings", { prompt: { client, file, replace } }).then(loadSettings);
  };
  document.querySelectorAll("[data-prompt-client]").forEach((el) =>
    el.addEventListener("change", () => save(el.dataset.promptClient)));
  document.querySelectorAll("[data-prompt-block]").forEach((el) =>
    el.addEventListener("change", () => save(el.dataset.promptBlock)));

  document.querySelectorAll("[data-loop-client]").forEach((el) =>
    el.addEventListener("change", () => post("/api/settings", {
      loops: { client: el.dataset.loopClient, count: Number(el.value) || 0 },
    }).then(loadSettings)));

  document.querySelectorAll("[data-toolset-client]").forEach((el) =>
    el.addEventListener("change", () => post("/api/settings", {
      toolset: { client: el.dataset.toolsetClient, file: el.value },
    }).then(loadSettings)));
}

async function loadSettings() {
  const data = await get("/api/settings");
  $("settings-path").textContent = "stored in " + data.path;
  $("params").innerHTML = data.params.map((p) => `
    <label class="flex cursor-pointer items-start gap-3 px-4 py-3">
      <input type="checkbox" data-param="${p.key}" ${p.stripped ? "checked" : ""}
             class="mt-0.5 h-4 w-4 shrink-0 accent-sky-600">
      <span class="min-w-0">
        <span class="font-mono text-sm ${p.stripped
          ? "text-sky-700 dark:text-sky-400" : ""}">${p.key}</span>
        ${p.stripped ? '<span class="ml-2 rounded bg-sky-100 px-1.5 py-0.5 text-xs text-sky-800 dark:bg-sky-900/40 dark:text-sky-300">stripped</span>' : ""}
        <span class="mt-0.5 block text-xs leading-relaxed text-slate-500 dark:text-slate-400">${p.why}</span>
      </span>
    </label>`).join("");

  renderPromptConfig(data);

  document.querySelectorAll("[data-param]").forEach((box) => {
    box.addEventListener("change", async () => {
      const strip = [...document.querySelectorAll("[data-param]")]
        .filter((b) => b.checked).map((b) => b.dataset.param);
      await post("/api/settings", { strip });
      loadSettings();
    });
  });
}

// ---------------------------------------------------------------------------
// Clients: where to point a coding client, with VS Code's block ready to paste.
// The engine builds both (client_setup.py); this only shows them.
// ---------------------------------------------------------------------------
async function loadClients() {
  const data = await get("/api/clients");
  const endpoints = data.endpoints || [];
  const copyButton = (key) => `<button data-copy="${key}"
      class="rounded-md border border-slate-300 dark:border-slate-600
             bg-transparent px-2 py-1 text-xs">copy</button>`;

  $("client-endpoints").innerHTML = endpoints.map((e, i) => `
    <div class="rounded-xl border border-slate-200 dark:border-[#262c36]
                bg-white dark:bg-[#161b22] p-4">
      <div class="flex flex-wrap items-center gap-3">
        <span class="font-medium">${esc(e.label)}</span>
        <code class="break-all font-mono text-sm text-sky-700 dark:text-sky-400">${esc(e.base_url)}</code>
        ${copyButton(i + ":url")}
      </div>
      ${e.note ? `<p class="mt-2 max-w-3xl text-xs leading-relaxed text-slate-500 dark:text-slate-400">${esc(e.note)}</p>` : ""}
      <div class="mt-4 flex flex-wrap items-center gap-3">
        <span class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">VS Code</span>
        <code class="font-mono text-xs">github.copilot.chat.customModels</code>
        ${copyButton(i + ":vscode")}
      </div>
      <pre class="mt-2 overflow-x-auto whitespace-pre rounded-lg border border-slate-200
                  dark:border-[#262c36] bg-slate-50 dark:bg-[#161b22] p-3 font-mono text-xs">${
        esc(JSON.stringify(e.vscode, null, 2))}</pre>
    </div>`).join("") ||
    `<p class="text-sm text-slate-500 dark:text-slate-400">No client addresses configured.</p>`;

  document.querySelectorAll("[data-copy]").forEach((button) => {
    button.addEventListener("click", () => {
      const [index, what] = button.dataset.copy.split(":");
      const endpoint = endpoints[Number(index)];
      const text = what === "url" ? endpoint.base_url : JSON.stringify(endpoint.vscode, null, 2);
      // The clipboard API needs a secure context - https or localhost. Over
      // plain http by IP it is missing, so say so rather than do nothing.
      if (!navigator.clipboard) {
        button.textContent = "select to copy";
        return;
      }
      navigator.clipboard.writeText(text).then(() => {
        button.textContent = "copied";
        setTimeout(() => { button.textContent = "copy"; }, 1500);
      }, () => { button.textContent = "select to copy"; });
    });
  });
}

document.querySelectorAll("[data-tab]").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});
showTab("dashboard");


// ---------------------------------------------------------------------------
// One session, in detail.
//
// The turn-by-turn view is where a loop is actually visible: identical
// consecutive tool calls, context climbing, output tokens pinned to the same
// number. The summary above it is the same pooled arithmetic as the table.
// ---------------------------------------------------------------------------
let currentSession = null;

async function showSession(id) {
  currentSession = id;
  location.hash = "session/" + id;
  showPanel("session");

  const data = await get("/api/sessions/" + encodeURIComponent(id));
  const s = data.session;
  if (!s) {
    $("session-summary").innerHTML =
      `<p class="text-sm text-slate-500 dark:text-slate-400">Session not found.</p>`;
    $("session-turns").innerHTML = "";
    return;
  }

  const from = parseTs(s.started), to = parseTs(s.ended);
  const wall = from && to ? (to - from) / 1000 : null;
  $("session-summary").innerHTML = `
    <h2 class="flex flex-wrap items-center gap-3 text-lg font-semibold">${s.alias}
      ${clientBadge(s.client)}
      <span class="font-mono text-xs font-normal text-slate-500 dark:text-slate-400">${s.session_id}</span>
    </h2>
    <div class="mt-4 grid gap-5 sm:grid-cols-3 lg:grid-cols-6">
      ${stat("Turns", fmt.int(s.turns))}
      ${stat("tok/s", fmt.one(s.tps), "pooled")}
      ${stat("Output", fmt.int(s.output_tokens), "tokens")}
      ${stat("Tool calls", fmt.int(s.tool_calls))}
      ${stat("Peak context", fmt.int(s.peak_context))}
      ${stat("Elapsed", wall === null ? "\u2014" : fmt.secs(wall), `${fmt.secs(s.model_s)} in the model`)}
    </div>`;

  let prev = null, repeats = 0;
  $("session-turns").innerHTML = data.turns.map((t, i) => {
    const key = (t.tool_names || "") + "|" + (t.tool_args || "");
    const repeat = key !== "|" && key === prev;
    repeats = repeat ? repeats + 1 : 0;
    prev = key;
    const flag = t.preempted ? "preempted"
      : !t.ok ? `error ${t.status || ""}`
      : t.finish_reason === "length" ? "truncated" : t.finish_reason || "";
    return `<tr data-turn="${t.id}"
      class="cursor-pointer border-b border-slate-100 dark:border-[#262c36]
             last:border-0 hover:bg-slate-50 dark:hover:bg-slate-800/50 ${
      repeat ? "bg-amber-50 dark:bg-amber-950/30" : ""}">
      <td class="px-3 py-2 font-mono text-xs text-slate-400">${i + 1}</td>
      <td class="px-3 py-2 font-mono text-xs">${fmt.time(t.ts)}</td>
      <td class="px-3 py-2 text-right tabular-nums">${fmt.int(t.output_tokens)}</td>
      <td class="px-3 py-2 text-right tabular-nums">${fmt.int(t.context_tokens)}</td>
      <td class="px-3 py-2 text-right tabular-nums">${fmt.one(t.decode_tps)}</td>
      <td class="px-3 py-2 text-right tabular-nums ${
        t.ttft_ms > 60000 ? "text-amber-600 dark:text-amber-400 font-medium" : ""
      }">${fmt.ms(t.ttft_ms)}</td>
      <td class="px-3 py-2 text-right tabular-nums ${
        t.deferral_ms > 10000 ? "text-amber-600 dark:text-amber-400" : ""
      }">${t.deferral_ms ? fmt.ms(t.deferral_ms) : "\u2014"}</td>
      <td class="px-3 py-2 font-mono text-xs">${t.tool_names || "\u2014"}${
        repeat ? `<span class="ml-2 text-amber-600 dark:text-amber-400">repeat \u00d7${repeats + 1}</span>` : ""}</td>
      <td class="px-3 py-2 text-xs ${
        t.preempted || !t.ok || t.finish_reason === "length"
          ? "text-amber-600 dark:text-amber-400" : "text-slate-500 dark:text-slate-400"
      }">${flag}</td>
    </tr>`;
  }).join("");
  bindTurnRows();
}


// ---------------------------------------------------------------------------
// One turn.
//
// Only what this turn added, not the whole conversation. Every request carries
// the entire history - 256 messages by turn 18 of a real session - so the unit
// worth reading is the tail beyond what the previous turn sent: the tool
// results that came back, and the reply they produced.
//
// The trace may be gone. It is gitignored, rotates per engine run, and
// --no-trace turns it off, so "not there" is an ordinary answer and the row's
// metrics are shown either way.
// ---------------------------------------------------------------------------
let currentTurn = null;

const ROLE_STYLE = {
  user:      "border-sky-400 dark:border-sky-600",
  assistant: "border-emerald-400 dark:border-emerald-600",
  tool:      "border-slate-300 dark:border-slate-600",
  system:    "border-violet-400 dark:border-violet-600",
};

function messageBlock(m) {
  const calls = (m.tool_calls || []).map((c) => `
    <div class="mt-2 rounded bg-slate-100 dark:bg-[#0d1117] p-2 font-mono text-xs">
      <span class="text-sky-700 dark:text-sky-400">${c.name}</span>(<span
        class="whitespace-pre-wrap break-all">${esc(c.arguments)}</span>)${
        c.clipped ? '<span class="text-amber-600 dark:text-amber-400"> …clipped</span>' : ""}
    </div>`).join("");
  return `<div class="border-l-2 ${ROLE_STYLE[m.role] || ROLE_STYLE.tool} pl-3 py-1">
    <div class="text-xs uppercase tracking-wide text-slate-500 dark:text-slate-400">${m.role}</div>
    ${m.content ? `<pre class="mt-1 whitespace-pre-wrap break-words text-sm">${esc(m.content)}${
      m.clipped ? '<span class="text-amber-600 dark:text-amber-400"> …clipped</span>' : ""}</pre>` : ""}
    ${calls}
  </div>`;
}

const esc = (t) => String(t == null ? "" : t)
  .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

async function showTurn(id) {
  currentTurn = id;
  location.hash = "turn/" + id;
  showPanel("turn");

  const data = await get("/api/turn/" + encodeURIComponent(id));
  const t = data.turn, tr = data.trace;
  if (!t) {
    $("turn-body").innerHTML = `<p class="text-sm">No such turn.</p>`;
    return;
  }

  $("turn-position").textContent =
    `turn ${t.turn_number} of ${t.session_turns}`;
  $("turn-prev").disabled = !t.prev_id;
  $("turn-next").disabled = !t.next_id;
  $("turn-prev").onclick = () => t.prev_id && showTurn(t.prev_id);
  $("turn-next").onclick = () => t.next_id && showTurn(t.next_id);

  const flags = [];
  if (t.preempted) flags.push("preempted");
  if (t.finish_reason === "length") flags.push("truncated");
  if (!t.ok) flags.push(`error ${t.status || ""}`);

  const metrics = `
    <div class="mb-5 grid gap-5 sm:grid-cols-3 lg:grid-cols-6">
      ${stat("Output", fmt.int(t.output_tokens), "tokens")}
      ${stat("Context", fmt.int(t.context_tokens), `${fmt.int(t.cached_tokens)} cached`)}
      ${stat("tok/s", fmt.one(t.decode_tps), t.rate_source || "")}
      ${stat("TTFT", fmt.ms(t.ttft_ms), t.deferral_ms ? `${fmt.ms(t.deferral_ms)} deferral` : "")}
      ${stat("Wall", fmt.ms(t.total_ms))}
      ${stat("Finish", t.finish_reason || "—", flags.join(", "))}
    </div>`;

  if (!tr.found) {
    $("turn-body").innerHTML = metrics + `
      <div class="rounded-xl border border-amber-300 dark:border-amber-800
                  bg-amber-50 dark:bg-amber-950/30 p-4 text-sm">
        <div class="font-medium">The trace for this turn is not available</div>
        <div class="mt-1 text-slate-600 dark:text-slate-400">${esc(tr.reason)}</div>
        <div class="mt-2 text-xs text-slate-500 dark:text-slate-400">
          The measurements above come from the database and are unaffected.</div>
      </div>`;
    return;
  }

  const notes = [];
  if (tr.stripped_params) notes.push(`stripped ${tr.stripped_params.join(", ")}`);
  if (tr.rewrote_model) notes.push(`model rewritten to ${tr.rewrote_model[1]}`);
  if (tr.repaired_tool_args) notes.push(`repaired ${tr.repaired_tool_args.join(", ")} arguments`);

  $("turn-body").innerHTML = metrics + `
    <div class="mb-3 flex flex-wrap gap-3 text-xs text-slate-500 dark:text-slate-400">
      <span>${tr.tools.length} tools offered</span>
      <span>${Object.entries(tr.params).map(([k, v]) => `${k}=${v}`).join(" ") || "no sampling params sent"}</span>
      ${notes.map((n) => `<span class="text-amber-600 dark:text-amber-400">${esc(n)}</span>`).join("")}
    </div>

    <h3 class="mb-2 text-sm font-semibold uppercase tracking-wide
               text-slate-500 dark:text-slate-400">New this turn</h3>
    <div class="mb-6 space-y-3 rounded-xl border border-slate-200 dark:border-[#262c36]
                bg-white dark:bg-[#161b22] p-4">
      ${tr.skipped ? `<p class="text-xs text-slate-500 dark:text-slate-400">
        ${tr.skipped} earlier messages in this request are not shown — they are the
        conversation so far, resent unchanged.</p>` : ""}
      ${tr.new_messages.map(messageBlock).join("") ||
        '<p class="text-sm text-slate-500 dark:text-slate-400">Nothing new — the request repeated the previous one.</p>'}
    </div>

    <h3 class="mb-2 text-sm font-semibold uppercase tracking-wide
               text-slate-500 dark:text-slate-400">Reply</h3>
    <div class="space-y-3 rounded-xl border border-slate-200 dark:border-[#262c36]
                bg-white dark:bg-[#161b22] p-4">
      ${tr.reasoning ? `<details class="text-sm">
        <summary class="cursor-pointer text-xs uppercase tracking-wide
                        text-slate-500 dark:text-slate-400">Reasoning</summary>
        <pre class="mt-2 whitespace-pre-wrap break-words text-sm
                    text-slate-600 dark:text-slate-400">${esc(tr.reasoning)}</pre>
      </details>` : ""}
      ${tr.reply ? messageBlock(tr.reply) :
        '<p class="text-sm text-slate-500 dark:text-slate-400">No reply recorded.</p>'}
      ${tr.error ? `<div class="text-sm text-rose-600 dark:text-rose-400">${esc(JSON.stringify(tr.error))}</div>` : ""}
    </div>`;
}

$("turn-back").addEventListener("click", () => {
  const back = currentSession;
  currentTurn = null;
  if (back) showSession(back); else { location.hash = ""; showTab("dashboard"); }
});

$("session-back").addEventListener("click", () => {
  currentSession = null;
  location.hash = "";
  showTab("dashboard");
});

window.addEventListener("hashchange", routeFromHash);

function routeFromHash() {
  const turn = location.hash.match(/^#turn\/(\d+)$/);
  if (turn) return showTurn(Number(turn[1]));
  const m = location.hash.match(/^#session\/(.+)$/);
  if (m) showSession(decodeURIComponent(m[1]));
  else if (currentSession) { currentSession = null; showTab("dashboard"); }
}

$("chat-form").addEventListener("submit", sendChat);

$("unload").addEventListener("click", async () => {
  await post("/api/unload");
  refresh();
});

// Last, after every declaration: routeFromHash() reads `currentSession`,
// and a `let` read before its declaration throws rather than returning
// undefined - which would abort the whole script silently.
routeFromHash();
refresh();
setInterval(refresh, POLL_MS);
