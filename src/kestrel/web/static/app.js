"use strict";

// ---------------------------------------------------------------- helpers

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key === "class") el.className = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (value === true) el.setAttribute(key, "");
    else if (value !== false && value != null) el.setAttribute(key, value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

const $ = (selector) => document.querySelector(selector);

function pretty(raw) {
  try { return JSON.stringify(typeof raw === "string" ? JSON.parse(raw) : raw, null, 2); } catch { return String(raw); }
}

// Minimal Markdown -> DOM. It builds nodes (never HTML strings), so model output cannot inject markup.
const INLINE = /(`[^`\n]+`|\*\*[^*\n]+\*\*|\[[^\]\n]+\]\(https?:\/\/[^)\s]+\))/g;

function renderInline(text, parent) {
  for (const part of text.split(INLINE)) {
    if (!part) continue;
    if (part.startsWith("`") && part.endsWith("`") && part.length > 1) parent.append(h("code", {}, part.slice(1, -1)));
    else if (part.startsWith("**") && part.endsWith("**") && part.length > 3) parent.append(h("strong", {}, part.slice(2, -2)));
    else if (part.startsWith("[") && part.includes("](")) {
      const split = part.lastIndexOf("](");
      parent.append(h("a", { href: part.slice(split + 2, -1), target: "_blank", rel: "noopener noreferrer" }, part.slice(1, split)));
    } else parent.append(document.createTextNode(part));
  }
}

function renderBlocks(text, parent) {
  for (const block of text.split(/\n{2,}/)) {
    const lines = block.split("\n").filter((l) => l.trim() !== "");
    if (!lines.length) continue;
    if (lines.every((l) => /^\s*[-*] /.test(l))) {
      const ul = h("ul");
      lines.forEach((l) => { const li = h("li"); renderInline(l.replace(/^\s*[-*] /, ""), li); ul.append(li); });
      parent.append(ul);
    } else if (lines.every((l) => /^\s*\d+[.)] /.test(l))) {
      const ol = h("ol");
      lines.forEach((l) => { const li = h("li"); renderInline(l.replace(/^\s*\d+[.)] /, ""), li); ol.append(li); });
      parent.append(ol);
    } else if (/^#{1,4} /.test(lines[0])) {
      const heading = h("p", {}, h("strong"));
      renderInline(lines[0].replace(/^#{1,4} /, ""), heading.firstChild);
      parent.append(heading);
      if (lines.length > 1) renderBlocks(lines.slice(1).join("\n"), parent);
    } else {
      const p = h("p");
      lines.forEach((l, i) => { if (i) p.append(h("br")); renderInline(l, p); });
      parent.append(p);
    }
  }
}

function renderMarkdown(text, into) {
  const frag = document.createDocumentFragment();
  text.split("```").forEach((part, i) => {
    if (i % 2 === 1) {
      const newline = part.indexOf("\n");
      const hasLang = newline >= 0 && /^[\w+#.-]*$/.test(part.slice(0, newline).trim());
      frag.append(h("pre", {}, h("code", {}, (hasLang ? part.slice(newline + 1) : part).replace(/\n$/, ""))));
    } else renderBlocks(part, frag);
  });
  into.replaceChildren(frag);
}

// ---------------------------------------------------------------- api

const TOKEN_KEY = "kestrel_token";
(function captureToken() {
  const fromHash = new URLSearchParams(location.hash.slice(1)).get("token");
  if (fromHash) {
    sessionStorage.setItem(TOKEN_KEY, fromHash);
    history.replaceState(null, "", location.pathname); // keep the token out of the address bar and history
  }
})();

function showGate(text) {
  if (text) $("#gate-text").textContent = text;
  $("#gate").hidden = false;
}

async function api(path, options = {}) {
  const headers = { "x-kestrel-token": sessionStorage.getItem(TOKEN_KEY) || "" };
  if (options.body !== undefined) headers["content-type"] = "application/json";
  const response = await fetch(path, { ...options, headers, body: options.body === undefined ? undefined : JSON.stringify(options.body) });
  if (response.status === 401) { showGate("The access token is missing or wrong. Open the full link printed by `kestrel web`."); throw new Error("unauthorized"); }
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}

// ---------------------------------------------------------------- state

const state = { sessions: [], current: null, running: false, info: null, abort: null, tab: "chat" };

// ---------------------------------------------------------------- sessions

async function loadSessions() {
  state.sessions = await api("/api/sessions");
  const list = $("#session-list");
  list.replaceChildren(...state.sessions.map((s) =>
    h("li", { class: s.id === state.current ? "active" : "", onclick: () => selectSession(s.id) },
      h("span", { class: "title" }, s.id.startsWith("tg:") ? h("span", { class: "badge" }, "TG") : null, s.id.startsWith("tg:") ? " " : null, s.title || "New chat", s.running ? h("span", { class: "dot" }) : null),
      h("span", { class: "meta" }, `${s.messages}`))));
}

async function newChat() {
  if (state.running) return;
  const { id } = await api("/api/sessions", { method: "POST", body: {} });
  await selectSession(id);
}

async function selectSession(id) {
  if (state.running && id !== state.current) return;
  state.current = id;
  await loadSessions();
  const data = await api(`/api/sessions/${id}/messages`);
  renderHistory(data.messages);
  if (state.tab === "trace") loadTrace();
  $("#input").focus();
}

// ---------------------------------------------------------------- chat rendering

const messagesEl = () => $("#messages");

function stickToBottom(fn) {
  const el = messagesEl();
  const near = el.scrollHeight - el.scrollTop - el.clientHeight < 140;
  fn();
  if (near) el.scrollTop = el.scrollHeight;
}

function bubble(role, text, extraClass = "") {
  const body = h("div", { class: `bubble ${role === "user" ? "user" : ""}` });
  if (role === "user") body.textContent = text; else renderMarkdown(text, body);
  const wrap = h("div", { class: `msg ${extraClass}` }, h("div", { class: "who" }, role === "user" ? "You" : "Kestrel"), body);
  return { wrap, body };
}

function toolCard(name, args, risk) {
  const status = h("span", { class: "status" }, "running…");
  const result = h("pre", {}, h("code", {}, "…"));
  const el = h("details", { class: "card" },
    h("summary", {}, h("strong", {}, name), risk ? h("span", { class: `badge ${risk}` }, risk) : null, status),
    h("div", { class: "body" }, h("div", { class: "label" }, "arguments"), h("pre", {}, h("code", {}, pretty(args))), h("div", { class: "label" }, "result"), result));
  return { el, status, result };
}

function finishCard(card, text, isError) {
  card.status.textContent = isError ? "error" : "done";
  card.status.className = `status ${isError ? "err" : "ok"}`;
  card.result.firstChild.textContent = text;
  if (isError) card.el.open = true;
}

function riskOf(name) {
  const tool = (state.info?.tools || []).find((t) => t.name === name);
  return tool ? tool.risk : null;
}

function renderHistory(messages) {
  const container = messagesEl();
  container.replaceChildren();
  const cards = new Map();
  for (const m of messages) {
    const cls = m.compacted ? "compacted" : "";
    if (m.role === "summary") {
      container.append(h("details", { class: "card" }, h("summary", {}, `Earlier conversation summarised (through message ${m.covers_through})`),
        h("div", { class: "body" }, h("p", {}, m.content))));
    } else if (m.role === "user") {
      container.append(bubble("user", m.content, cls).wrap);
    } else if (m.role === "assistant") {
      if (m.content) container.append(bubble("assistant", m.content, cls).wrap);
      for (const call of m.tool_calls || []) {
        const card = toolCard(call.name, call.arguments, riskOf(call.name));
        cards.set(call.id, card);
        container.append(card.el);
      }
    } else if (m.role === "tool") {
      const card = cards.get(m.tool_call_id);
      if (card) finishCard(card, m.content, /^(Denied|Invalid arguments|Unknown tool|Outcome unknown)|failed:|timed out/.test(m.content));
    }
  }
  if (!messages.length) container.append(h("div", { class: "note" }, "Say hello, ask a question, or ask Kestrel to work with files in its workspace."));
  container.scrollTop = container.scrollHeight;
}

// ---------------------------------------------------------------- sending / streaming

function setRunning(running) {
  state.running = running;
  $("#send").hidden = running;
  $("#stop").hidden = !running;
  $("#input").disabled = running;
  $("#new-chat").disabled = running;
  if (!running) $("#input").focus();
}

async function send(text) {
  if (state.running || !text.trim()) return;
  if (!state.current) await newChat();
  const container = messagesEl();
  if (container.querySelector(".note") && !container.querySelector(".msg")) container.replaceChildren();
  stickToBottom(() => container.append(bubble("user", text).wrap));
  setRunning(true);
  state.abort = new AbortController();

  const live = { bubble: null, text: "", frame: 0, cards: new Map() };
  const repaint = () => {
    live.frame = 0;
    stickToBottom(() => renderMarkdown(live.text, live.bubble.body));
  };
  const handle = (ev) => {
    switch (ev.type) {
      case "token":
        if (!live.bubble) { live.bubble = bubble("assistant", ""); live.text = ""; stickToBottom(() => container.append(live.bubble.wrap)); }
        live.text += ev.text;
        if (!live.frame) live.frame = requestAnimationFrame(repaint);
        break;
      case "tool_call": {
        live.bubble = null;
        const card = toolCard(ev.name, ev.arguments, riskOf(ev.name));
        live.cards.set(ev.id, card);
        stickToBottom(() => container.append(card.el));
        break;
      }
      case "tool_result": {
        const card = live.cards.get(ev.id);
        if (card) finishCard(card, ev.text, ev.is_error);
        break;
      }
      case "approval_request": stickToBottom(() => container.append(approvalCard(ev))); break;
      case "approval_expired": document.querySelectorAll(`[data-approval="${ev.id}"]`).forEach((el) => markDecided(el, "expired")); break;
      case "compaction": stickToBottom(() => container.append(h("div", { class: "note" }, "Context was compacted to stay within the model's window."))); break;
      case "error": stickToBottom(() => container.append(h("div", { class: "note error" }, `Error: ${ev.message}`))); break;
      case "cancelled": stickToBottom(() => container.append(h("div", { class: "note" }, "Stopped."))); break;
      case "done": {
        if (live.frame) { cancelAnimationFrame(live.frame); repaint(); }
        const bits = [`${ev.steps} model call${ev.steps === 1 ? "" : "s"}`, `${ev.tokens} tokens`];
        if (ev.status !== "completed") bits.unshift(ev.status.replace("_", " "));
        stickToBottom(() => container.append(h("div", { class: "meta-line" }, bits.join(" · "))));
        if (ev.status !== "completed" && ev.error) stickToBottom(() => container.append(h("div", { class: "note error" }, ev.error)));
        break;
      }
    }
  };

  try {
    const response = await fetch(`/api/sessions/${state.current}/messages`, {
      method: "POST", signal: state.abort.signal,
      headers: { "x-kestrel-token": sessionStorage.getItem(TOKEN_KEY) || "", "content-type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (response.status === 401) { showGate("The access token is missing or wrong."); return; }
    if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.error || `HTTP ${response.status}`); }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let split;
      while ((split = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        if (chunk.startsWith("data: ")) handle(JSON.parse(chunk.slice(6)));
      }
    }
  } catch (error) {
    if (error.name !== "AbortError") stickToBottom(() => container.append(h("div", { class: "note error" }, `Error: ${error.message}`)));
  } finally {
    state.abort = null;
    setRunning(false);
    loadSessions().catch(() => {});
  }
}

async function stop() {
  if (!state.current) return;
  api(`/api/sessions/${state.current}/stop`, { method: "POST", body: {} }).catch(() => {});
}

function markDecided(el, text) {
  el.classList.add("decided");
  el.querySelectorAll("button").forEach((b) => (b.disabled = true));
  const label = el.querySelector(".verdict");
  if (label) label.textContent = text;
}

function approvalCard(ev) {
  const verdict = h("span", { class: "verdict hint" }, "waiting for you…");
  const decide = async (allow) => {
    markDecided(card, allow ? "allowed" : "denied");
    try { await api(`/api/approvals/${ev.id}`, { method: "POST", body: { allow } }); } catch (e) { verdict.textContent = e.message; }
  };
  const card = h("div", { class: "approval-card", "data-approval": ev.id },
    h("strong", {}, "Approval needed: "), h("code", {}, ev.tool), " ", h("span", { class: `badge ${ev.risk}` }, ev.risk),
    h("pre", {}, h("code", {}, pretty(ev.args))),
    h("div", { class: "actions" }, h("button", { class: "primary", onclick: () => decide(true) }, "Allow"),
      h("button", { onclick: () => decide(false) }, "Deny"), verdict));
  return card;
}

// ---------------------------------------------------------------- trace

async function loadTrace() {
  if (!state.current) return;
  const { events, summary } = await api(`/api/sessions/${state.current}/trace`);
  const stat = (n, l) => h("div", { class: "stat" }, h("div", { class: "n" }, n), h("div", { class: "l" }, l));
  const price = summary.cost > 0 ? stat(`$${summary.cost.toFixed(4)}`, "estimated cost") : null;
  $("#trace-summary").replaceChildren(
    stat(summary.llm_calls, "model calls"), stat(`${(summary.llm_ms / 1000).toFixed(1)}s`, "model time"),
    stat(summary.tool_calls, "tool calls"), stat(summary.tool_errors, "tool errors"),
    stat(summary.prompt_tokens, "prompt tokens"), stat(summary.completion_tokens, "completion tokens"), ...(price ? [price] : []),
    ...Object.entries(summary.recoveries).map(([name, n]) => stat(n, `recovery: ${name}`)));
  const t0 = events.length ? events[0].ts : 0;
  const rows = events.map((e) => {
    const d = e.data || {};
    let detail = "";
    if (e.kind === "llm") detail = `in ${d.prompt_tokens ?? 0} / out ${d.completion_tokens ?? 0} · ${d.finish_reason ?? ""}${d.tool_calls?.length ? " · " + d.tool_calls.join(", ") : ""}`;
    else if (e.kind === "tool") detail = d.preview ? String(d.preview).slice(0, 120) : "";
    else detail = Object.entries(d).filter(([k]) => k !== "error").map(([k, v]) => `${k}=${v}`).join(" ");
    if (d.error) detail += ` !! ${d.error}`;
    const bad = d.is_error || d.error;
    return h("tr", { class: bad ? "err" : "" }, h("td", { class: "num" }, `+${(e.ts - t0).toFixed(2)}s`), h("td", {}, e.kind), h("td", {}, e.name),
      h("td", { class: "num" }, Math.round(e.duration_ms)), h("td", {}, detail));
  });
  $("#trace-table tbody").replaceChildren(...rows);
}

// ---------------------------------------------------------------- memory

async function loadMemory() {
  const data = await api("/api/memory");
  $("#persona").value = data.persona;
  $("#facts-usage").textContent = `(${data.used_chars}/${data.max_chars} characters)`;
  renderFacts(data.facts);
}

function renderFacts(facts) {
  const list = $("#facts");
  if (!facts.length) { list.replaceChildren(h("li", { class: "empty" }, "Nothing remembered yet. Add a fact, or tell Kestrel “remember that …”.")); return; }
  list.replaceChildren(...facts.map((fact) => h("li", {}, h("span", {}, fact),
    h("button", { onclick: async () => { const r = await api("/api/memory/facts", { method: "DELETE", body: { text: fact } }); renderFacts(r.facts); } }, "Remove"))));
}

async function searchMemory(query) {
  const out = $("#search-results");
  out.replaceChildren(h("div", { class: "empty" }, "Searching…"));
  try {
    const { hits } = await api(`/api/memory/search?q=${encodeURIComponent(query)}`);
    out.replaceChildren(...(hits.length ? hits.map((x) => h("div", { class: "hit" },
      h("div", { class: "meta" }, `${x.age} · session ${x.session_id} · fused ${x.score.toFixed(3)} · dense ${x.dense.toFixed(2)} · bm25 ${x.sparse.toFixed(2)}`), x.text))
      : [h("div", { class: "empty" }, "No relevant past conversations.")]));
  } catch (error) { out.replaceChildren(h("div", { class: "note error" }, error.message)); }
}

// ---------------------------------------------------------------- tools & skills

function renderTools() {
  const info = state.info;
  if (!info) return;
  $("#tools-table tbody").replaceChildren(...info.tools.map((t) => h("tr", {}, h("td", {}, h("code", {}, t.name)), h("td", {}, h("span", { class: `badge ${t.risk}` }, t.risk)),
    h("td", {}, [t.idempotent ? "idempotent" : null, t.parallel_safe ? "parallel" : null, t.external ? "mcp" : null].filter(Boolean).join(", ")), h("td", {}, t.description))));
  $("#skills-list").replaceChildren(...(info.skills.length ? info.skills.map((s) => h("div", { class: "skill" },
    h("strong", {}, s.name), " ", h("span", { class: "badge" }, s.source), s.available ? null : h("span", { class: "badge dangerous" }, `missing ${s.missing.join(", ")}`), h("div", {}, s.description)))
    : [h("div", { class: "empty" }, "No skills found. Add a folder with a SKILL.md to <workspace>/skills/.")]));
  const servers = Object.entries(info.mcp);
  $("#mcp-list").replaceChildren(...(servers.length ? servers.map(([name, status]) => h("div", { class: "server" }, h("strong", {}, name), " ",
    h("span", { class: `badge ${status === "ok" ? "read" : "dangerous"}` }, status === "ok" ? "connected" : "failed"), status === "ok" ? null : h("div", {}, status)))
    : [h("div", { class: "empty" }, "No MCP servers configured. Start with --mcp servers.json.")]));
}

// ---------------------------------------------------------------- sub-agents

let jobsTimer = 0;

async function loadJobs() {
  const jobs = await api("/api/jobs");
  $("#jobs-list").replaceChildren(...(jobs.length ? jobs.map((j) => h("div", { class: "job" },
    h("div", { class: "meta" }, `${j.job_id} · ${j.profile} · parent ${j.parent} · ${j.status}${j.error ? " · " + j.error : ""}`),
    h("strong", {}, j.label || j.task.slice(0, 80)),
    j.status === "running" ? h("div", {}, h("button", { onclick: async () => { await api(`/api/jobs/${j.job_id}/cancel`, { method: "POST", body: {} }).catch(() => {}); loadJobs(); } }, "Cancel")) : null,
    j.result ? h("pre", {}, h("code", {}, j.result)) : null))
    : [h("div", { class: "empty" }, "No sub-agent jobs yet. Ask Kestrel to delegate research or scripting work.")]));
}

// ---------------------------------------------------------------- tabs & boot

function switchTab(name) {
  state.tab = name;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".tab").forEach((s) => s.classList.toggle("active", s.id === `tab-${name}`));
  clearInterval(jobsTimer);
  if (name === "trace") loadTrace();
  if (name === "memory") loadMemory();
  if (name === "tools") renderTools();
  if (name === "agents") { loadJobs(); jobsTimer = setInterval(loadJobs, 3000); }
}

function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 200) + "px";
}

async function boot() {
  if (!sessionStorage.getItem(TOKEN_KEY)) { showGate(); return; }
  try {
    state.info = await api("/api/state");
  } catch (error) { if (error.message !== "unauthorized") showGate(error.message); return; }
  const info = state.info;
  $("#approval-mode").value = info.approval;
  $("#model-info").textContent = `${info.model} · embeddings: ${info.embeddings} · workspace: ${info.workspace}`;
  document.title = `Kestrel · ${info.model}`;
  await loadSessions();
  if (state.sessions.length) await selectSession(state.sessions[0].id); else await newChat();
}

$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const input = $("#input"); const text = input.value; input.value = ""; autoGrow(input); send(text); });
$("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#input").addEventListener("input", (e) => autoGrow(e.target));
$("#stop").addEventListener("click", stop);
$("#new-chat").addEventListener("click", () => newChat());
$("#tabs").addEventListener("click", (e) => { if (e.target.dataset.tab) switchTab(e.target.dataset.tab); });
$("#trace-refresh").addEventListener("click", loadTrace);
$("#jobs-refresh").addEventListener("click", loadJobs);
$("#approval-mode").addEventListener("change", async (e) => {
  try { await api("/api/approval-mode", { method: "POST", body: { mode: e.target.value } }); } catch (error) { alert(error.message); e.target.value = state.info.approval; return; }
  state.info.approval = e.target.value;
});
$("#persona-save").addEventListener("click", async () => {
  const status = $("#persona-status");
  try { await api("/api/memory/persona", { method: "PUT", body: { text: $("#persona").value } }); status.textContent = "Saved."; } catch (error) { status.textContent = error.message; }
});
$("#fact-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("#fact-input");
  if (!input.value.trim()) return;
  try { const r = await api("/api/memory/facts", { method: "POST", body: { fact: input.value } }); renderFacts(r.facts); input.value = ""; loadMemory(); } catch (error) { alert(error.message); }
});
$("#search-form").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#search-input").value.trim(); if (q) searchMemory(q); });

boot();
