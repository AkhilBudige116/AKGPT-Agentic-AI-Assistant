/* AKGPT front-end (vanilla JS). All model output is Markdown -> marked -> DOMPurify -> DOM;
   user/tool text is only ever inserted with textContent. No secrets exist in this file. */
(() => {
  "use strict";

  // ───────────────────────────── helpers ─────────────────────────────
  const $ = (sel, root = document) => root.querySelector(sel);
  const make = (tag, attrs = {}, ...kids) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (v === undefined || v === null || v === false) continue;
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid) node.append(kid);
    return node;
  };
  const uuid = () => (crypto.randomUUID ? crypto.randomUUID() :
    "10000000-1000-4000-8000-100000000000".replace(/[018]/g, (c) => (c ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (c / 4)))).toString(16)));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
    del(k) { try { localStorage.removeItem(k); } catch { /* ignore */ } },
  };

  class ApiError extends Error {
    constructor(message, status, code) { super(message); this.status = status; this.code = code; }
  }

  async function api(path, { method = "GET", json, form, signal } = {}) {
    const res = await fetch(path, {
      method, signal, credentials: "same-origin",
      headers: json !== undefined ? { "Content-Type": "application/json" } : {},
      body: json !== undefined ? JSON.stringify(json) : form,
    });
    if (res.status === 204) return null;
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON body */ }
    if (!res.ok) throw new ApiError((data && data.error) || `Request failed (${res.status})`, res.status, data && data.code);
    return data;
  }

  // ───────────────────────────── state ─────────────────────────────
  const state = {
    threadId: null, title: "New chat", conversations: [], messages: [], documents: [], uploads: [],
    settings: null, models: [], streaming: false, abort: null, stick: true, userPersisted: false,
    lastRequest: null, theme: "system",
  };

  const dom = {
    app: $("#app"), sidebar: $("#sidebar"), backdrop: $("#backdrop"), convList: $("#conv-list"), convEmpty: $("#conv-empty"),
    messages: $("#messages"), wrap: $("#messages-wrap"), empty: $("#empty-state"), jump: $("#jump-bottom"),
    title: $("#chat-title"), model: $("#model-select"), composer: $("#composer"), input: $("#composer-input"),
    send: $("#send"), stop: $("#stop"), voice: $("#voice-input"), file: $("#file-input"), attach: $("#attach"), docList: $("#doc-list"),
    drop: $("#drop-overlay"), sr: $("#sr-status"), toasts: $("#toasts"), exportBtn: $("#export-chat"), clearBtn: $("#clear-chat"),
    themeLabel: $("#theme-label"), themeIcon: $("#theme-icon"),
  };
  let recognition = null;
  let dictating = false;

  const TOOL_LABEL = {
    web_search: "Searching the web", get_weather: "Checking the weather", calculator: "Calculating",
    search_uploaded_documents: "Searching your documents", remember_information: "Saving to memory",
    recall_memory: "Recalling memory", search_conversation_history: "Searching this chat",
    update_memory: "Updating memory", forget_memory: "Forgetting a memory",
  };
  const TOOL_DONE = {
    web_search: "Web search", get_weather: "Weather", calculator: "Calculator", search_uploaded_documents: "Documents",
    remember_information: "Memory saved", recall_memory: "Memory", search_conversation_history: "Chat history",
    update_memory: "Memory updated", forget_memory: "Memory removed",
  };

  // ───────────────────────────── toasts / dialogs ─────────────────────────────
  function toast(message, kind = "info", ms = 5000) {
    const t = make("div", { class: `toast ${kind === "error" ? "error" : ""}`, role: "status", text: message });
    dom.toasts.append(t);
    setTimeout(() => t.remove(), ms);
  }
  function confirmDialog(title, text, okLabel = "Confirm") {
    const dlg = $("#confirm-dialog");
    $("#confirm-title").textContent = title;
    $("#confirm-text").textContent = text;
    $("#confirm-ok").textContent = okLabel;
    return new Promise((resolve) => {
      dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok"), { once: true });
      dlg.returnValue = "cancel";
      dlg.showModal();
    });
  }
  function promptRename(current) {
    const dlg = $("#rename-dialog"), input = $("#rename-input");
    input.value = current;
    return new Promise((resolve) => {
      dlg.addEventListener("close", () => resolve(dlg.returnValue === "ok" ? input.value.trim() : null), { once: true });
      dlg.returnValue = "cancel";
      dlg.showModal();
      input.select();
    });
  }

  // ───────────────────────────── markdown ─────────────────────────────
  marked.setOptions({ gfm: true, breaks: true });
  DOMPurify.addHook("afterSanitizeAttributes", (node) => {
    if (node.tagName === "A") {
      node.setAttribute("target", "_blank");
      node.setAttribute("rel", "noopener noreferrer nofollow");
    }
  });
  function renderMarkdown(text, target) {
    const html = DOMPurify.sanitize(marked.parse(text || ""), {
      FORBID_TAGS: ["style", "form", "input", "button", "textarea", "select", "iframe", "object", "embed"],
      FORBID_ATTR: ["style", "srcset"],
    });
    target.innerHTML = html;
    for (const pre of [...target.querySelectorAll("pre")]) {
      const code = pre.querySelector("code");
      const lang = ((code && [...code.classList].find((c) => c.startsWith("language-"))) || "").replace("language-", "");
      const wrapper = make("div", { class: "codeblock" },
        make("div", { class: "code-head" }, make("span", { text: lang || "code" }),
          make("button", { class: "copy-code", type: "button", text: "Copy", "aria-label": "Copy code" })));
      pre.replaceWith(wrapper);
      wrapper.append(pre);
    }
  }
  async function copyText(text) {
    try { await navigator.clipboard.writeText(text); return true; } catch {
      const ta = make("textarea", { "aria-hidden": "true" });
      ta.value = text; document.body.append(ta); ta.select();
      let ok = false;
      try { ok = document.execCommand("copy"); } catch { ok = false; }
      ta.remove();
      return ok;
    }
  }
  dom.messages.addEventListener("click", async (e) => {
    const btn = e.target.closest(".copy-code");
    if (!btn) return;
    const pre = btn.closest(".codeblock").querySelector("pre");
    if (await copyText(pre.innerText)) { btn.textContent = "Copied"; setTimeout(() => (btn.textContent = "Copy"), 1500); }
    else toast("Could not copy to clipboard.", "error");
  });

  // ───────────────────────────── scrolling ─────────────────────────────
  dom.wrap.addEventListener("scroll", () => {
    const near = dom.wrap.scrollHeight - dom.wrap.scrollTop - dom.wrap.clientHeight < 90;
    state.stick = near;
    dom.jump.hidden = near;
  });
  dom.jump.addEventListener("click", () => { state.stick = true; scrollBottom(true); });
  function scrollBottom(force = false) {
    if (force || state.stick) dom.wrap.scrollTop = dom.wrap.scrollHeight;
  }

  // ───────────────────────────── message rendering ─────────────────────────────
  function sourceChips(sources) {
    if (!sources || !sources.length) return null;
    const row = make("div", { class: "sources" }, make("span", { class: "label", text: "Sources:" }));
    for (const s of sources.slice(0, 12)) {
      const title = String(s.title || s.url || "source");
      if (typeof s.url === "string" && /^https?:\/\//i.test(s.url)) {
        row.append(make("a", { class: "source-chip", href: s.url, target: "_blank", rel: "noopener noreferrer nofollow", title: s.url, text: title }));
      } else {
        row.append(make("span", { class: "source-chip", text: title }));
      }
    }
    return row;
  }
  function weatherCard(d) {
    if (!d || d.temperature === undefined) return null;
    const loc = d.location || {}, u = d.units || {};
    const place = [loc.name, loc.country].filter(Boolean).join(", ");
    const num = (v, unit = "") => (v === undefined || v === null ? "—" : `${Math.round(v * 10) / 10}${unit}`);
    return make("div", { class: "weather", role: "group", "aria-label": `Current weather in ${place}` },
      make("div", { class: "w-temp", text: num(d.temperature, u.temperature || "") }),
      make("div", { class: "w-place", text: place || "Weather" }),
      make("div", { class: "w-desc", text: d.description || "" }),
      make("div", { class: "w-meta" },
        make("span", { text: `Feels like ${num(d.feels_like, u.temperature || "")}` }),
        make("span", { text: `Humidity ${num(d.humidity, "%")}` }),
        make("span", { text: `Wind ${num(d.wind_speed, " " + (u.wind_speed || ""))}` }),
        d.observed_at_utc ? make("span", { text: `Observed ${new Date(d.observed_at_utc).toLocaleString()}` }) : null));
  }
  function toolChip(t, running) {
    const cls = running ? "running" : t.ok ? "ok" : "fail";
    const label = running ? (TOOL_LABEL[t.name] || t.name) + "…" : (TOOL_DONE[t.name] || t.name) + (t.ok ? "" : " failed");
    return make("span", { class: `tool-chip ${cls}`, "data-tool": t.name, "data-id": t.id || "", text: label });
  }
  function metaLine(m) {
    const bits = [];
    if (m.model) bits.push(m.model);
    if (m.latency_ms) bits.push(`${(m.latency_ms / 1000).toFixed(1)}s`);
    const tokens = (m.prompt_tokens || 0) + (m.completion_tokens || 0);
    if (tokens) bits.push(`${tokens} tokens`);
    return bits.join(" · ");
  }

  function buildMessage(msg, isLast) {
    const isUser = msg.role === "user";
    const root = make("article", { class: `msg ${isUser ? "user" : "assistant"}${isLast ? " last" : ""}`, "data-id": msg.id || "" });
    const body = make("div", { class: "msg-body" });
    const content = make("div", { class: "content" });
    msg.nodes = { root, body, content };
    if (!isUser) {
      root.append(make("div", { class: "avatar", "aria-hidden": "true", text: "AK" }));
      const tools = make("div", { class: "tools" });
      msg.nodes.tools = tools;
      body.append(tools);
      for (const t of (msg.meta && msg.meta.tools) || []) {
        tools.append(toolChip(t, false));
        if (t.name === "get_weather" && t.ok) { const w = weatherCard(t.data); if (w) body.append(w); }
      }
      if (!tools.children.length) tools.hidden = true;
    }
    body.append(content);
    if (isUser) content.textContent = msg.content; else renderMarkdown(msg.content, content);
    if (!isUser) {
      msg.nodes.sources = make("div");
      body.append(msg.nodes.sources);
      const s = sourceChips(msg.meta && msg.meta.sources);
      if (s) msg.nodes.sources.append(s);
      if (msg.meta && msg.meta.interrupted) body.append(make("div", { class: "msg-meta", text: "Generation was stopped." }));
      const ml = metaLine(msg);
      msg.nodes.metaLine = make("div", { class: "msg-meta", text: ml });
      if (!ml) msg.nodes.metaLine.hidden = true;
      body.append(msg.nodes.metaLine);
    }
    msg.nodes.actions = make("div", { class: "msg-actions" });
    body.append(msg.nodes.actions);
    fillActions(msg, isLast);
    root.append(body);
    return root;
  }
  function fillActions(msg, isLast) {
    const bar = msg.nodes.actions;
    bar.replaceChildren();
    if (msg.pending) return;
    bar.append(make("button", { class: "mini", type: "button", text: "Copy", "aria-label": "Copy message",
      onclick: async (e) => { const ok = await copyText(msg.content); e.target.textContent = ok ? "Copied" : "Failed"; setTimeout(() => (e.target.textContent = "Copy"), 1500); } }));
    if (msg.role === "assistant" && isLast && !state.streaming) {
      bar.append(make("button", { class: "mini", type: "button", text: "↻ Regenerate", "aria-label": "Regenerate answer", onclick: () => regenerate() }));
    }
    if (msg.role === "user" && isLast && !state.streaming) {
      bar.append(make("button", { class: "mini", type: "button", text: "✎ Edit", "aria-label": "Edit and resend", onclick: () => beginEdit(msg) }));
    }
  }
  function renderAll() {
    dom.messages.querySelectorAll(".msg").forEach((n) => n.remove());
    dom.empty.hidden = state.messages.length > 0;
    const lastAssistantOrUser = state.messages.length - 1;
    state.messages.forEach((m, i) => dom.messages.append(buildMessage(m, i === lastAssistantOrUser)));
    dom.title.textContent = state.title;
    dom.exportBtn.disabled = dom.clearBtn.disabled = !state.threadId || state.messages.length === 0;
    scrollBottom(true);
  }
  function refreshLast() {
    const msgs = state.messages;
    msgs.forEach((m, i) => { if (m.nodes) { m.nodes.root.classList.toggle("last", i === msgs.length - 1); fillActions(m, i === msgs.length - 1); } });
  }

  // ───────────────────────────── conversations ─────────────────────────────
  async function loadConversations() {
    const data = await api("/api/conversations");
    state.conversations = Array.isArray(data && data.conversations) ? data.conversations : [];
    renderConvList();
  }
  function renderConvList() {
    dom.convList.replaceChildren();
    dom.convEmpty.hidden = state.conversations.length > 0;
    for (const c of state.conversations) {
      const li = make("li", { class: `conv-item${c.thread_id === state.threadId ? " active" : ""}` },
        make("button", { class: "conv-open", type: "button", text: c.title, title: c.title, "aria-current": c.thread_id === state.threadId ? "true" : null,
          onclick: () => { openConversation(c.thread_id); closeSidebar(); } }),
        make("span", { class: "conv-actions" },
          make("button", { class: "icon-btn", type: "button", text: "✎", title: "Rename", "aria-label": `Rename ${c.title}`, onclick: () => renameConversation(c) }),
          make("button", { class: "icon-btn", type: "button", text: "🗑", title: "Delete", "aria-label": `Delete ${c.title}`, onclick: () => deleteConversation(c) })));
      dom.convList.append(li);
    }
  }
  async function openConversation(id) {
    if (state.streaming) return;
    try {
      const d = await api(`/api/conversations/${id}`);
      state.threadId = d.thread_id; state.title = d.title;
      state.messages = (d.messages || []).map((m) => ({ id: m.message_id, role: m.role, content: m.content, meta: m.meta || {}, model: m.model,
        latency_ms: m.latency_ms, prompt_tokens: m.prompt_tokens, completion_tokens: m.completion_tokens }));
      state.documents = d.documents || [];
      store.set("akgpt.thread", state.threadId);
      renderAll(); renderDocs(); renderConvList();
    } catch (e) {
      if (e.status === 404) { store.del("akgpt.thread"); newChat(); } else toast(e.message, "error");
    }
  }
  function newChat() {
    if (state.streaming) return;
    state.threadId = null; state.title = "New chat"; state.messages = []; state.documents = [];
    store.del("akgpt.thread");
    renderAll(); renderDocs(); renderConvList();
    dom.input.focus();
  }
  async function renameConversation(c) {
    const title = await promptRename(c.title);
    if (!title || title === c.title) return;
    try {
      await api(`/api/conversations/${c.thread_id}`, { method: "PATCH", json: { title } });
      if (c.thread_id === state.threadId) { state.title = title; dom.title.textContent = title; }
      await loadConversations();
    } catch (e) { toast(e.message, "error"); }
  }
  async function deleteConversation(c) {
    if (!(await confirmDialog("Delete conversation?", `“${c.title}” and its uploaded documents will be permanently deleted.`, "Delete"))) return;
    try {
      await api(`/api/conversations/${c.thread_id}`, { method: "DELETE" });
      if (c.thread_id === state.threadId) { state.streaming = false; newChat(); }
      await loadConversations();
      toast("Conversation deleted.");
    } catch (e) { toast(e.message, "error"); }
  }

  // ───────────────────────────── chat / SSE ─────────────────────────────
  function setStreaming(on) {
    state.streaming = on;
    dom.send.hidden = on; dom.stop.hidden = !on;
    dom.input.disabled = false;
    dom.sr.textContent = on ? "AKGPT is responding" : "Response complete";
    refreshLast();
  }
  function parseFrames(buffer) {
    const frames = []; let idx;
    while ((idx = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, idx); buffer = buffer.slice(idx + 2);
      let event = "message"; const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith(":")) continue;
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
      }
      if (data.length) { try { frames.push([event, JSON.parse(data.join("\n"))]); } catch { /* ignore malformed frame */ } }
    }
    return [frames, buffer];
  }

  async function send(text, { mode = "send" } = {}) {
    if (state.streaming) return;
    text = (text || "").trim();
    if (mode !== "regenerate" && !text) return;
    const clientId = mode === "send" ? uuid() : undefined;
    state.lastRequest = { mode, text };
    state.userPersisted = false;

    // optimistic UI
    if (mode === "send") {
      state.messages.push({ id: clientId, role: "user", content: text });
    } else {
      const last = state.messages[state.messages.length - 1];
      if (last && last.role === "assistant") state.messages.pop();
      if (mode === "edit") {
        const lastUser = state.messages[state.messages.length - 1];
        if (lastUser && lastUser.role === "user") lastUser.content = text;
      }
    }
    const reply = { id: null, role: "assistant", content: "", meta: {}, pending: true };
    state.messages.push(reply);
    renderAll();
    reply.nodes.root.classList.add("streaming");
    reply.nodes.content.replaceChildren(make("span", { class: "typing", "aria-label": "Thinking" }, make("span"), make("span"), make("span")));
    state.stick = true; scrollBottom(true);

    const body = { message: mode === "regenerate" ? "regenerate" : text, mode, model: dom.model.value || undefined };
    if (state.threadId) body.thread_id = state.threadId;
    if (clientId) body.client_message_id = clientId;

    state.abort = new AbortController();
    setStreaming(true);
    let raw = "", frameQueued = false, finished = false, started = false;
    const paint = () => {
      frameQueued = false;
      if (!started) { reply.nodes.content.replaceChildren(); started = true; }
      renderMarkdown(raw, reply.nodes.content);
      scrollBottom();
    };
    const schedule = () => { if (!frameQueued) { frameQueued = true; requestAnimationFrame(paint); } };

    const onEvent = (name, d) => {
      if (name === "meta") {
        state.userPersisted = true;
        const isNew = !state.threadId;
        state.threadId = d.thread_id; state.title = d.title || state.title; dom.title.textContent = state.title;
        store.set("akgpt.thread", state.threadId);
        dom.exportBtn.disabled = dom.clearBtn.disabled = false;
        if (isNew) loadConversations().catch(() => {});
      } else if (name === "delta") {
        raw += d.text; schedule();
      } else if (name === "reset") {
        raw = ""; schedule();
      } else if (name === "tool_start") {
        reply.nodes.tools.hidden = false;
        reply.nodes.tools.append(toolChip({ name: d.name, id: d.id }, true));
      } else if (name === "tool_end") {
        const running = [...reply.nodes.tools.querySelectorAll(".tool-chip.running")].find((c) => c.dataset.tool === d.name);
        const chip = toolChip({ name: d.name, ok: d.ok }, false);
        if (running) running.replaceWith(chip); else reply.nodes.tools.append(chip);
        if (d.name === "get_weather" && d.ok) { const w = weatherCard(d.data); if (w) reply.nodes.body.insertBefore(w, reply.nodes.content); }
        if (!d.ok && d.error) toast(d.error, "error", 7000);
      } else if (name === "done") {
        finished = true;
        if (d.status === "error") return;
        reply.id = d.message_id; reply.pending = false; reply.content = d.content || raw;
        reply.model = d.model; reply.latency_ms = d.latency_ms;
        reply.prompt_tokens = d.usage && d.usage.input_tokens; reply.completion_tokens = d.usage && d.usage.output_tokens;
        reply.meta = { tools: d.tools || [], sources: d.sources || [] };
        raw = reply.content; paint();
        const src = sourceChips(d.sources);
        reply.nodes.sources.replaceChildren(...(src ? [src] : []));
        const ml = metaLine(reply); reply.nodes.metaLine.textContent = ml; reply.nodes.metaLine.hidden = !ml;
        if (d.title) { state.title = d.title; dom.title.textContent = d.title; }
        loadConversations().catch(() => {});
      } else if (name === "error") {
        showError(reply, d.error || "Something went wrong.");
      }
    };

    try {
      const res = await fetch("/api/chat/stream", {
        method: "POST", credentials: "same-origin", signal: state.abort.signal,
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify(body),
      });
      if (!res.ok) {
        let msg = `Request failed (${res.status})`;
        try { msg = (await res.json()).error || msg; } catch { /* ignore */ }
        throw new ApiError(msg, res.status);
      }
      const reader = res.body.getReader(), decoder = new TextDecoder();
      let buf = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
        const [frames, rest] = parseFrames(buf); buf = rest;
        for (const [n, d] of frames) onEvent(n, d);
      }
      if (!finished) showError(reply, "The connection ended before the answer was complete.");
    } catch (e) {
      if (e.name === "AbortError") {
        reply.pending = false; reply.content = raw;
        reply.nodes.body.append(make("div", { class: "msg-meta", text: "Generation was stopped." }));
        if (!raw) reply.nodes.content.replaceChildren();
      } else {
        showError(reply, e.message || "Network error.");
      }
    } finally {
      reply.nodes.root.classList.remove("streaming");
      if (reply.pending) { reply.pending = false; reply.content = raw; }
      if (!reply.content && !reply.nodes.body.querySelector(".error-box") && reply.nodes.content.querySelector(".typing")) {
        reply.nodes.content.replaceChildren();
      }
      state.abort = null;
      setStreaming(false);
      dom.input.focus();
    }
  }
  function showError(reply, message) {
    reply.pending = false;
    reply.nodes.content.querySelector(".typing")?.remove();
    reply.nodes.body.querySelector(".error-box")?.remove();
    const box = make("div", { class: "error-box", role: "alert" }, make("span", { text: message }),
      make("button", { class: "btn btn-primary", type: "button", text: "Retry", onclick: () => retryLast() }));
    reply.nodes.body.insertBefore(box, reply.nodes.actions);
  }
  function retryLast() {
    if (state.streaming || !state.lastRequest) return;
    const last = state.messages[state.messages.length - 1];
    if (last && last.role === "assistant" && !last.content) state.messages.pop();
    const { mode, text } = state.lastRequest;
    if (state.userPersisted || mode !== "send") {
      // the server already stored the user message - answer it again
      if (mode === "send") state.messages.pop(); // drop the optimistic user bubble; the server re-adds it
      return mode === "edit" ? send(text, { mode: "edit" }) : (state.messages.length && state.messages[state.messages.length - 1].role !== "user" ? send(text, { mode: "send" }) : regenerateAfterError(text, mode));
    }
    state.messages.pop(); // optimistic user bubble was never stored
    return send(text, { mode: "send" });
  }
  function regenerateAfterError(text, mode) {
    if (mode === "send") return send(text, { mode: "regenerate" });
    return send(text, { mode });
  }
  function regenerate() { if (state.threadId && !state.streaming) send("regenerate", { mode: "regenerate" }); }
  function beginEdit(msg) {
    if (state.streaming) return;
    const ta = make("textarea", { "aria-label": "Edit your message" });
    ta.value = msg.content;
    const box = make("div", { class: "edit-box" }, ta, make("div", { class: "edit-actions" },
      make("button", { class: "btn btn-ghost", type: "button", text: "Cancel", onclick: () => renderAll() }),
      make("button", { class: "btn btn-primary", type: "button", text: "Save & resend", onclick: () => { const v = ta.value.trim(); if (v) send(v, { mode: "edit" }); } })));
    msg.nodes.content.replaceChildren(box); msg.nodes.actions.replaceChildren();
    ta.focus();
  }

  dom.stop.addEventListener("click", () => state.abort && state.abort.abort());
  function updateVoiceButton(active) {
    dictating = active;
    dom.voice.classList.toggle("recording", active);
    dom.voice.setAttribute("aria-pressed", String(active));
    dom.voice.setAttribute("aria-label", active ? "Stop voice input" : "Start voice input");
    dom.voice.title = active ? "Stop voice input" : "Start voice input";
    dom.voice.textContent = active ? "■" : "🎙";
  }
  dom.voice.addEventListener("click", () => {
    if (dictating) {
      recognition.stop();
      updateVoiceButton(false);
      dom.sr.textContent = "Voice input stopped.";
      return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      toast("Voice input is not supported in this browser. Try Chrome or Edge.", "error");
      return;
    }

    if (!recognition) {
      recognition = new SpeechRecognition();
      recognition.lang = navigator.language || "en-US";
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.onresult = (event) => {
        let transcript = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          if (event.results[i].isFinal) transcript += event.results[i][0].transcript;
        }
        if (transcript) {
          const separator = dom.input.value && !/\s$/.test(dom.input.value) ? " " : "";
          dom.input.value += separator + transcript.trim();
          autosize();
          dom.input.focus();
        }
      };
      recognition.onerror = (event) => {
        updateVoiceButton(false);
        if (event.error === "not-allowed" || event.error === "service-not-allowed") {
          toast("Microphone access was denied. Allow microphone access in your browser settings.", "error");
        } else if (event.error === "network") {
          toast("Voice recognition needs a network connection. Check your connection and try again.", "error");
        } else if (event.error !== "aborted" && event.error !== "no-speech") {
          toast("Voice input stopped unexpectedly. Please try again.", "error");
        }
      };
      recognition.onend = () => {
        if (dictating) {
          updateVoiceButton(false);
          dom.sr.textContent = "Voice input stopped.";
        }
      };
    }

    try {
      recognition.start();
      updateVoiceButton(true);
      dom.sr.textContent = "Listening. Speak to enter your message.";
    } catch {
      updateVoiceButton(false);
      toast("Could not start voice input. Please try again.", "error");
    }
  });
  window.addEventListener("pagehide", () => {
    if (dictating && recognition) recognition.stop();
  });
  dom.composer.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = dom.input.value;
    if (!text.trim() || state.streaming) return;
    if (dictating) {
      recognition.stop();
      updateVoiceButton(false);
    }
    dom.input.value = ""; autosize();
    send(text);
  });
  dom.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); dom.composer.requestSubmit(); }
  });
  function autosize() { dom.input.style.height = "auto"; dom.input.style.height = Math.min(dom.input.scrollHeight, 220) + "px"; }
  dom.input.addEventListener("input", autosize);
  document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => { dom.input.value = b.dataset.prompt; autosize(); dom.input.focus(); }));

  // ───────────────────────────── documents ─────────────────────────────
  async function loadDocs() {
    if (!state.threadId) { state.documents = []; return renderDocs(); }
    try { state.documents = (await api(`/api/documents?thread_id=${state.threadId}`)).documents || []; } catch { /* ignore */ }
    renderDocs();
  }
  function renderDocs() {
    dom.docList.replaceChildren();
    for (const u of state.uploads) {
      dom.docList.append(make("li", { class: "doc" }, make("span", { class: "name", text: u.name }),
        u.phase === "uploading" ? make("progress", { max: "100", value: String(u.pct), "aria-label": `Uploading ${u.name}` }) : null,
        make("span", { class: "status", text: u.phase === "uploading" ? `${u.pct}%` : "indexing…" })));
    }
    for (const d of state.documents) {
      dom.docList.append(make("li", { class: `doc ${d.indexing_status}` },
        make("span", { class: "name", title: d.filename, text: d.filename }),
        make("span", { class: "status", text: d.indexing_status === "indexed" ? `✓ ${d.chunk_count} chunks` : d.indexing_status === "failed" ? (d.error || "failed") : d.indexing_status }),
        make("button", { class: "icon-btn", type: "button", text: "✕", title: "Remove document", "aria-label": `Remove ${d.filename}`, onclick: () => removeDoc(d) })));
    }
    dom.docList.hidden = dom.docList.children.length === 0;
  }
  async function removeDoc(d) {
    if (!(await confirmDialog("Remove document?", `“${d.filename}” and its indexed content will be deleted.`, "Remove"))) return;
    try { await api(`/api/documents/${d.document_id}`, { method: "DELETE" }); await loadDocs(); } catch (e) { toast(e.message, "error"); }
  }
  function uploadOne(file) {
    return new Promise((resolve, reject) => {
      const entry = { name: file.name, pct: 0, phase: "uploading" };
      state.uploads.push(entry); renderDocs();
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/upload");
      xhr.upload.onprogress = (e) => { if (e.lengthComputable) { entry.pct = Math.round((e.loaded / e.total) * 100); renderDocs(); } };
      xhr.upload.onload = () => { entry.phase = "indexing"; renderDocs(); };
      const done = (fn) => { state.uploads = state.uploads.filter((u) => u !== entry); renderDocs(); fn(); };
      xhr.onerror = () => done(() => reject(new Error("Upload failed — network error.")));
      xhr.onload = () => {
        let data = null; try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
        done(() => (xhr.status >= 200 && xhr.status < 300 && data ? resolve(data) : reject(new Error((data && data.error) || `Upload failed (${xhr.status})`))));
      };
      const fd = new FormData();
      fd.append("file", file);
      if (state.threadId) fd.append("thread_id", state.threadId);
      xhr.send(fd);
    });
  }
  async function uploadFiles(files) {
    const exts = (state.settings && state.settings.supported_extensions) || [".pdf", ".docx", ".txt", ".md", ".csv", ".py"];
    const maxMb = (state.settings && state.settings.max_upload_mb) || 20;
    for (const file of files) {
      const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
      if (!exts.includes(ext)) { toast(`${file.name}: unsupported type. Use ${exts.join(", ")}.`, "error"); continue; }
      if (file.size > maxMb * 1024 * 1024) { toast(`${file.name}: larger than ${maxMb} MB.`, "error"); continue; }
      try {
        const d = await uploadOne(file);
        if (!state.threadId) { state.threadId = d.thread_id; store.set("akgpt.thread", d.thread_id); await loadConversations(); }
        toast(d.duplicate ? `${d.filename} was already uploaded.` : `Indexed ${d.filename} (${d.chunk_count} chunks).`);
      } catch (e) { toast(`${file.name}: ${e.message}`, "error", 8000); }
    }
    await loadDocs(); await loadConversations();
    renderConvList();
  }
  dom.attach.addEventListener("click", () => dom.file.click());
  dom.file.addEventListener("change", () => { const f = [...dom.file.files]; dom.file.value = ""; if (f.length) uploadFiles(f); });
  let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { dragDepth++; dom.drop.hidden = false; } });
  window.addEventListener("dragleave", (e) => { if (hasFiles(e) && --dragDepth <= 0) { dragDepth = 0; dom.drop.hidden = true; } });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("drop", (e) => { if (!hasFiles(e)) return; e.preventDefault(); dragDepth = 0; dom.drop.hidden = true; uploadFiles([...e.dataTransfer.files]); });

  // ───────────────────────────── theme / settings ─────────────────────────────
  const prefersDark = () => window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  function applyTheme(theme) {
    state.theme = theme;
    document.documentElement.setAttribute("data-theme", theme);
    store.set("akgpt.theme", theme);
    const dark = theme === "dark" || (theme === "system" && prefersDark());
    dom.themeIcon.textContent = dark ? "☀" : "☾";
    dom.themeLabel.textContent = dark ? "Light mode" : "Dark mode";
  }
  $("#toggle-theme").addEventListener("click", async () => {
    const dark = state.theme === "dark" || (state.theme === "system" && prefersDark());
    const next = dark ? "light" : "dark";
    applyTheme(next);
    try { await api("/api/settings", { method: "POST", json: { theme: next } }); } catch { /* theme still applied locally */ }
  });

  function fillModels(select, selected) {
    select.replaceChildren(...state.models.map((m) => make("option", { value: m, text: m })));
    if (selected) select.value = selected;
  }
  async function loadSettings() {
    state.settings = await api("/api/settings");
    state.models = state.settings.available_models || [];
    fillModels(dom.model, state.settings.model);
    applyTheme(state.settings.theme || "system");
  }
  dom.model.addEventListener("change", async () => {
    try { await api("/api/settings", { method: "POST", json: { model: dom.model.value } }); } catch (e) { toast(e.message, "error"); }
  });

  const sd = $("#settings-dialog");
  const tempIn = $("#set-temp"), topkIn = $("#set-topk");
  tempIn.addEventListener("input", () => ($("#set-temp-out").textContent = Number(tempIn.value).toFixed(2)));
  topkIn.addEventListener("input", () => ($("#set-topk-out").textContent = topkIn.value));
  async function loadMemories() {
    const list = $("#memory-list"); list.replaceChildren();
    try {
      const mems = (await api("/api/memories")).memories || [];
      $("#memory-empty").hidden = mems.length > 0;
      for (const m of mems) {
        list.append(make("li", {}, make("span", { text: m.content }),
          make("button", { class: "icon-btn", type: "button", text: "✕", title: "Forget this", "aria-label": `Forget: ${m.content}`,
            onclick: async () => { try { await api(`/api/memories/${m.memory_id}`, { method: "DELETE" }); loadMemories(); } catch (e) { toast(e.message, "error"); } } })));
      }
    } catch (e) { toast(e.message, "error"); }
  }
  $("#open-settings").addEventListener("click", async () => {
    try { await loadSettings(); } catch (e) { toast(e.message, "error"); return; }
    const s = state.settings;
    fillModels($("#set-model"), s.model);
    tempIn.value = s.temperature; $("#set-temp-out").textContent = Number(s.temperature).toFixed(2);
    topkIn.value = s.top_k; $("#set-topk-out").textContent = s.top_k;
    $("#set-style").value = s.response_style; $("#set-theme").value = s.theme;
    $("#mode-notice").textContent = `Mode: ${s.deployment_mode}.`;
    const on = Object.entries(s.integrations).map(([k, v]) => `${k}: ${v ? "configured" : "not configured"}`).join(" · ");
    $("#integrations").textContent = `Integrations — ${on}`;
    loadMemories();
    sd.returnValue = "cancel"; sd.showModal();
  });
  $("#settings-form").addEventListener("submit", async () => {
    if (sd.returnValue !== "save") return;
    try {
      await api("/api/settings", { method: "POST", json: {
        model: $("#set-model").value, temperature: Number(tempIn.value), top_k: Number(topkIn.value),
        response_style: $("#set-style").value, theme: $("#set-theme").value } });
      await loadSettings(); toast("Settings saved.");
    } catch (e) { toast(e.message, "error"); }
  });
  // Make sure the clicked submit button's value is what ends up as returnValue.
  $("#save-settings").addEventListener("click", () => { sd.returnValue = "save"; });

  // ───────────────────────────── misc UI ─────────────────────────────
  function openSidebar() { dom.sidebar.classList.add("open"); dom.backdrop.hidden = false; }
  function closeSidebar() { dom.sidebar.classList.remove("open"); dom.backdrop.hidden = true; }
  $("#open-sidebar").addEventListener("click", openSidebar);
  $("#close-sidebar").addEventListener("click", closeSidebar);
  dom.backdrop.addEventListener("click", closeSidebar);
  $("#new-chat").addEventListener("click", () => { newChat(); closeSidebar(); });
  dom.exportBtn.addEventListener("click", () => { if (state.threadId) window.location.assign(`/api/conversations/${state.threadId}/export?format=md`); });
  dom.clearBtn.addEventListener("click", async () => {
    if (!state.threadId || !(await confirmDialog("Clear conversation?", "All messages in this conversation will be deleted. Uploaded documents are kept.", "Clear"))) return;
    try {
      await api(`/api/conversations/${state.threadId}/messages`, { method: "DELETE" });
      state.messages = []; state.title = "New chat"; renderAll(); await loadConversations(); toast("Conversation cleared.");
    } catch (e) { toast(e.message, "error"); }
  });
  window.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });
  if (window.matchMedia) window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => applyTheme(state.theme));

  // ───────────────────────────── boot ─────────────────────────────
  (async function init() {
    applyTheme(store.get("akgpt.theme") || "system");
    try {
      await loadSettings();
      await loadConversations();
      const saved = store.get("akgpt.thread");
      if (saved && state.conversations.some((c) => c.thread_id === saved)) await openConversation(saved);
      else renderAll();
    } catch (e) { toast(e.message || "Could not reach the server.", "error", 8000); renderAll(); }
    dom.input.focus();
  })();
})();
