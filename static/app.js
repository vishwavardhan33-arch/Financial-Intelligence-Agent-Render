(() => {
  "use strict";

  const form = document.getElementById("ask-form");
  const input = document.getElementById("question");
  const button = document.getElementById("ask-button");
  const log = document.getElementById("log");
  const starter = document.getElementById("starter");
  const notice = document.getElementById("notice");
  const modelLine = document.getElementById("model-line");

  const PERCENT_INTENTS = new Set(["gross_margin", "operating_margin", "net_margin", "growth_rate", "cagr"]);
  const MONEY_COLUMN = /(amount|total|price|spend|invoiced|revenue|cost|balance)/i;

  /* ---------- tiny DOM helpers (all text goes through textContent) ---------- */

  function el(tag, props = {}, children = []) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value === undefined || value === null) continue;
      if (key === "class") node.className = value;
      else if (key === "text") node.textContent = value;
      else node.setAttribute(key, value);
    }
    for (const child of [].concat(children)) {
      if (child) node.append(child);
    }
    return node;
  }

  function svgMark(kind, index) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 18 18");
    svg.setAttribute("class", `mark ${kind}`);
    svg.setAttribute("aria-hidden", "true");
    svg.style.setProperty("--i", index);
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d",
      kind === "ok" ? "M2.5 9.5 L7 14 L15.5 3.5"
      : kind === "bad" ? "M3.5 3.5 L14.5 14.5 M14.5 3.5 L3.5 14.5"
      : "M4 9 L14 9");
    svg.append(path);
    return svg;
  }

  /* ---------- formatting ---------- */

  const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 });
  const moneyFmt = new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  function formatCell(column, value) {
    if (value === null || value === undefined) return "";
    if (typeof value === "number") return MONEY_COLUMN.test(column) ? moneyFmt.format(value) : numberFmt.format(value);
    return String(value);
  }

  function humanize(name) {
    const spaced = String(name).replace(/_/g, " ");
    return spaced.charAt(0).toUpperCase() + spaced.slice(1);
  }

  function formatCalcValue(intent, value) {
    if (typeof value !== "number") return String(value);
    if (PERCENT_INTENTS.has(intent)) return `${(value * 100).toLocaleString("en-US", { maximumFractionDigits: 2 })}%`;
    return numberFmt.format(value);
  }

  function formatInput(value) {
    return typeof value === "number" ? numberFmt.format(value) : String(value);
  }

  /* ---------- content builders ---------- */

  // Answers are plain text from the model. Honour **bold** and paragraph breaks, nothing else.
  function renderAnswer(text) {
    const wrap = el("div", { class: "answer" });
    for (const para of String(text).split(/\n{2,}/)) {
      const p = el("p");
      para.split(/(\*\*[^*]+\*\*)/g).forEach((piece) => {
        if (piece.startsWith("**") && piece.endsWith("**") && piece.length > 4) {
          p.append(el("strong", { text: piece.slice(2, -2) }));
        } else if (piece) {
          p.append(document.createTextNode(piece));
        }
      });
      wrap.append(p);
    }
    return wrap;
  }

  function rowsTable(rows) {
    const columns = Object.keys(rows[0]);
    const numeric = new Set(columns.filter((c) => rows.every((r) => r[c] === null || typeof r[c] === "number")));
    const head = el("tr", {}, columns.map((c) => el("th", { class: numeric.has(c) ? "num" : "", text: c })));
    const body = rows.map((row) =>
      el("tr", {}, columns.map((c) => el("td", { class: numeric.has(c) ? "num" : "", text: formatCell(c, row[c]) })))
    );
    return el("div", { class: "table-wrap" }, el("table", { class: "data" }, [el("thead", {}, head), el("tbody", {}, body)]));
  }

  function parseMarkdownTable(lines) {
    const cells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
    const rows = lines.filter((l) => !/^\|\s*:?-{2,}/.test(l.trim())).map(cells);
    if (!rows.length) return null;
    const [header, ...rest] = rows;
    const isNum = (c) => /^[\d,.\-()%$]+$/.test(c);
    const numericCols = new Set(header.map((_, i) => i).filter((i) => rest.length && rest.every((r) => isNum(r[i] || ""))));
    return el("div", { class: "table-wrap" }, el("table", { class: "data" }, [
      el("thead", {}, el("tr", {}, header.map((h, i) => el("th", { class: numericCols.has(i) ? "num" : "", text: h })))),
      el("tbody", {}, rest.map((r) => el("tr", {}, r.map((c, i) => el("td", { class: numericCols.has(i) ? "num" : "", text: c }))))),
    ]));
  }

  function passageBody(text) {
    const lines = String(text).split("\n");
    const tableLines = lines.filter((l) => l.trim().startsWith("|"));
    if (tableLines.length < 2) return [el("p", { class: "passage-text", text })];
    const prose = lines.filter((l) => !l.trim().startsWith("|")).join("\n").trim();
    const out = [];
    if (prose) out.push(el("p", { class: "passage-text", text: prose }));
    const table = parseMarkdownTable(tableLines);
    if (table) out.push(table);
    return out;
  }

  /* ---------- one step of the working ---------- */

  function stepView(step, result, index, citations) {
    const skipped = result === undefined;
    const failed = !skipped && result && result.step_error !== undefined;
    const toolName = { sql: "Invoice database", rag: "Filings", calc: "Calculation", custom_calc: "Calculation" }[step.tool] || step.tool;

    const margin = el("div", { class: "step-margin" }, [
      svgMark(skipped ? "skipped" : failed ? "bad" : "ok", index),
      el("span", { class: "tool", text: toolName }),
    ]);

    const content = el("div", { class: "step-content" });

    if (skipped) {
      content.append(el("p", { class: "step-title", text: "Not run" }));
      content.append(el("p", { class: "step-note", text: "An earlier step failed, so this one was not attempted." }));
    } else if (failed) {
      content.append(el("p", { class: "step-title", text: step.query || humanize(step.intent || step.expression || step.tool) }));
      content.append(el("p", { class: "step-error", text: result.step_error }));
    } else if (step.tool === "sql") {
      content.append(el("p", { class: "step-title", text: step.query }));
      const rows = result.rows || [];
      if (rows.length) {
        content.append(rowsTable(rows));
        content.append(el("p", { class: "row-count", text: `${rows.length} ${rows.length === 1 ? "row" : "rows"}` }));
      } else {
        content.append(el("p", { class: "step-note", text: "The query ran and returned no rows." }));
      }
      if (result.sql) {
        content.append(el("details", { class: "sql" }, [
          el("summary", { text: "SQL that ran" }),
          el("pre", { class: "code", text: result.sql }),
        ]));
      }
    } else if (step.tool === "calc" || step.tool === "custom_calc") {
      const intent = result.intent || step.intent;
      content.append(el("p", { class: "step-title", text: step.tool === "calc" ? humanize(intent) : "Custom calculation" }));
      content.append(el("p", { class: "result-value", text: formatCalcValue(intent, result.value) }));
      const used = result.inputs_used || result.values_used;
      if (used && Object.keys(used).length) {
        const dl = el("dl", { class: "inputs" });
        for (const [k, v] of Object.entries(used)) {
          dl.append(el("dt", { text: humanize(k) }), el("dd", { text: formatInput(v) }));
        }
        content.append(dl);
      }
      if (step.tool === "custom_calc" && result.expression) {
        content.append(el("details", { class: "sql" }, [
          el("summary", { text: "Formula" }),
          el("pre", { class: "code", text: result.expression }),
        ]));
      }
    } else if (step.tool === "rag") {
      content.append(el("p", { class: "step-title", text: step.query }));
      const chunks = result.chunks || [];
      if (!chunks.length) content.append(el("p", { class: "step-note", text: "No matching passages were found." }));
      chunks.forEach((chunk) => {
        const cited = citations.includes(chunk.chunk_id);
        const label = [`Passage ${chunk.chunk_id + 1}`, chunk.section, cited ? "used in the answer" : null].filter(Boolean).join(", ");
        content.append(el("div", { class: `passage${cited ? " is-cited" : ""}` }, [
          el("p", { class: "passage-label", text: label }),
          ...passageBody(chunk.text),
        ]));
      });
    }

    return el("li", { class: `step${skipped ? " is-skipped" : ""}` }, [margin, content]);
  }

  /* ---------- an entry in the log ---------- */

  function newEntry(question) {
    const body = el("div", { class: "entry-body" }, [el("h2", { class: "question", text: question })]);
    const gutter = el("div", { class: "entry-gutter", text: new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) });
    const entry = el("article", { class: "entry" }, [gutter, body]);
    return { entry, body };
  }

  function showPending(body) {
    const message = el("p", { text: "Planning the steps, then running them." });
    const pending = el("div", { class: "pending" }, [message, el("div", { class: "pen" })]);
    body.append(pending);
    const started = Date.now();
    const timer = setInterval(() => {
      const seconds = Math.round((Date.now() - started) / 1000);
      message.textContent = seconds >= 8
        ? "Still working. If the server was idle it can take up to a minute to wake up."
        : "Planning the steps, then running them.";
    }, 1000);
    return { pending, stop: () => { clearInterval(timer); pending.remove(); } };
  }

  function showFailure(body, headline, detail) {
    body.append(el("div", { class: "failure", role: "alert" }, [
      el("strong", { text: headline }),
      detail ? el("p", { text: detail }) : null,
    ]));
  }

  function showResult(body, data) {
    if (data.answer) body.append(renderAnswer(data.answer));
    if (data.error) {
      showFailure(body, data.answer ? "Part of this could not be completed" : "No answer this time", data.error);
    }

    const plan = data.plan || [];
    if (plan.length) {
      const list = el("ol", { class: "working", id: `working-${Math.random().toString(36).slice(2)}` });
      plan.forEach((step, i) => list.append(stepView(step, data.step_results[i], i, data.citations || [])));

      const toggle = el("button", { type: "button", class: "working-toggle", "aria-expanded": "true", text: "Hide working" });
      toggle.setAttribute("aria-controls", list.id);
      toggle.addEventListener("click", () => {
        const entry = body.closest(".entry");
        const collapsed = entry.classList.toggle("is-collapsed");
        toggle.textContent = collapsed ? "Show working" : "Hide working";
        toggle.setAttribute("aria-expanded", String(!collapsed));
      });
      body.append(toggle, list);
    }
  }

  function collapseOlderEntries() {
    log.querySelectorAll(".entry:not(.is-collapsed)").forEach((entry) => {
      if (entry === log.firstElementChild) return;
      entry.classList.add("is-collapsed");
      const toggle = entry.querySelector(".working-toggle");
      if (toggle) {
        toggle.textContent = "Show working";
        toggle.setAttribute("aria-expanded", "false");
      }
    });
  }

  /* ---------- asking ---------- */

  let busy = false;

  async function ask(question) {
    const text = question.trim();
    if (busy || text.length < 3) return;
    busy = true;
    button.disabled = true;
    starter.hidden = true;

    const { entry, body } = newEntry(text);
    log.prepend(entry);
    collapseOlderEntries();
    const pending = showPending(body);

    try {
      const response = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      pending.stop();

      if (response.status === 429) {
        const detail = (await response.json().catch(() => ({}))).detail;
        showFailure(body, "Slow down a little", detail || "Too many questions in a short time. Wait a minute and ask again.");
      } else if (response.status === 422) {
        showFailure(body, "That question can't be sent", "Questions need to be between 3 and 500 characters.");
      } else if (!response.ok) {
        showFailure(body, "The server hit a problem", `It answered with status ${response.status}. Try again in a moment.`);
      } else {
        showResult(body, await response.json());
      }
    } catch (err) {
      pending.stop();
      showFailure(body, "Could not reach the server", "Check your connection and ask again.");
    } finally {
      busy = false;
      button.disabled = false;
      input.value = "";
      input.focus();
    }
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    ask(input.value);
  });

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      form.requestSubmit();
    }
  });

  document.querySelectorAll("[data-q]").forEach((btn) => {
    btn.addEventListener("click", () => ask(btn.dataset.q));
  });

  /* ---------- server status ---------- */

  fetch("/api/status")
    .then((r) => (r.ok ? r.json() : null))
    .then((status) => {
      if (!status) return;
      modelLine.textContent = `Model: ${status.model}`;
      modelLine.hidden = false;
      if (!status.llm_configured) {
        notice.textContent = "This server has no LLM_API_KEY set, so questions can't be answered yet. Add a free key in the environment settings and redeploy.";
        notice.hidden = false;
      }
    })
    .catch(() => {});
})();
