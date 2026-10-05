# Financial Intelligence Agent

A deployable agent that answers financial questions by routing between:
- **RAG** over financial filings (10-K/10-Q/earnings reports)
- **SQL** queries against a normalized invoice database
- **Python calculations** (margins, growth, CAGR, ratios)

It ships with a web UI where every answer comes with its working: the SQL that ran,
the numbers it produced, the calculation, and the filing passages the answer relied on.

**Free to run.** The LLM is an open model (Llama 3.3 70B by default) served through
Groq's free API tier, via any OpenAI-compatible endpoint. Everything else is open source:
Postgres, BGE embeddings (ONNX), in-process Qdrant, LangGraph, FastAPI. Hosting fits Render's
free tier. Note that questions you ask are sent to the LLM provider.

## Architecture

```
User Query
   |
   v
Planner (LangGraph, structured JSON plan, pydantic-validated)
   |
   v
Execute steps in order (plan-then-execute) -- a failed step halts before
dependent steps run, e.g.:
   Step 0: SQL   -> LLM-generated SQL, sqlglot-validated, read-only exec
   Step 1: Calc  -> deterministic metric fn, called with Step 0's REAL values
                    (never an LLM-retyped number -- see agent/resolve.py)
   Step N: RAG   -> Qdrant hybrid search (dense+BM25, RRF) + reranking
   |
   v
Synthesizer -> final answer + structured citations
              (citations validated against real retrieved chunk_ids;
               fabricated ones are dropped, not trusted)
```

## Project layout

```
financial-intelligence-agent/
  db/
    schema.sql          # Postgres schema: vendors, invoices, line_items + agent_readonly role
    models.py             # SQLAlchemy ORM models matching schema.sql
    session.py              # DB engine/session factory (separate owner vs. read-only engines)
  ingestion/
    load_invoices.py    # CSV/Excel -> normalized Postgres tables
    sample_data/           # example CSV for testing the loader
  retrieval/
    parse_filing.py     # Docling: PDF -> structured Markdown (tables preserved)
    chunker.py             # hybrid chunking: structure-aware text + atomic table chunks
    embeddings.py             # FastEmbedder (ONNX, deployed) + BGEEmbedder (torch) + fake (tests)
    corpus.py                    # corpus.jsonl read/write (chunks + optional vectors)
    export_corpus.py              # CLI: PDFs/Markdown -> data/corpus.jsonl
    vector_store.py             # Qdrant hybrid search (dense+BM25, RRF) + reranking
    index_filings.py               # CLI: PDF dir -> parse -> chunk -> embed -> index
    tests/                            # offline via fake embedder/reranker
  sql_node/
    schema_context.py   # allow-listed schema description + few-shot NL->SQL examples
    prompts.py             # text-to-SQL prompt builder
    validator.py              # sqlglot validation: single SELECT, allow-listed tables/columns
    executor.py                  # runs validated SQL via the read-only role
    sql_node.py                     # orchestrates: prompt -> LLM SQL -> validate -> execute
    tests/                            # validator tests (offline) + integration (live DB)
  calc/
    metrics.py           # pure functions: margins, growth_rate, cagr, ratio
    deterministic.py        # intent -> function dispatch, no LLM involved
    safe_eval.py                # restricted AST expression evaluator (no eval()/exec())
    calc_node.py                    # hybrid orchestrator: deterministic + LLM-tool-calling paths
    tests/                             # unit tests + SQL-Node-to-Calc-Node chain integration
  agent/
    plan.py               # pydantic schema for the planner's structured multi-step output
    planner.py               # builds the planning prompt, parses+validates the LLM's plan
    resolve.py                  # resolves step-arg references to real prior-step outputs
    dispatch.py                    # executes one plan step (rag/sql/calc/custom_calc)
    synthesizer.py                    # final-answer prompt + structured citation validation
    graph.py                             # LangGraph wiring: plan -> execute (loop) -> synthesize
    tests/                                  # plan validation, resolution, full graph (live DB)
  evals/
    judge_prompts.py     # LLM-as-judge prompt templates (retrieval/SQL/end-to-end)
    judge.py                # judge harness: prompt -> LLM call -> structured score
    run_evals.py                # runs a query batch through the agent, then judges each
    tests/                         # judge parsing + full agent-to-judge pipeline (live DB)
  api/
    llm_provider.py      # OpenAI-compatible client (Groq default), retries on 429/5xx
    rag_setup.py            # lazy filings index built from data/corpus.jsonl
    main.py                    # FastAPI: POST /query, GET /health, GET /api/status, serves the UI
    tests/                         # TestClient tests, full flow against live DB
  static/                 # the web UI (plain HTML/CSS/JS, no build step)
  data/                     # corpus.jsonl (prebuilt filing chunks) + one sample filing
  db/init_db.py         # idempotent schema + seed, runs on every container start
  start.sh                   # container entrypoint: init DB, then uvicorn
  render.yaml            # Render Blueprint: free web service + free Postgres
  docker-compose.yml    # local Postgres + app
  Dockerfile
  requirements.txt            # runtime (light: no torch)
  requirements-ingest.txt  # only for parsing PDFs on your own machine (Docling, torch)
  .env.example
```

## What's been verified, not just written

- **DB + ingestion**: schema applies cleanly to a live Postgres; ingestion correctly groups
  line items into invoices; the `agent_readonly` role can `SELECT` but a `DELETE` is
  rejected at the database level.
- **Retrieval**: a synthetic filing with a real Markdown table chunks correctly — the table
  survives as one atomic, captioned chunk, never split mid-row. Hybrid search (dense+BM25
  RRF fusion), reranking, and metadata filtering all confirmed against an in-memory Qdrant
  instance, including a test proving a company-scoped query never leaks another company's data.
- **SQL Node**: valid joins/aggregates/CTEs execute against live Postgres and return real
  rows. A multi-statement injection attempt and a hallucinated column are both rejected
  *before* touching the database — verified by re-querying the DB afterward and confirming
  it's still intact. Two real validator bugs (a CTE alias and an ORDER BY alias both being
  misidentified as unknown columns/tables) were caught by testing and fixed.
- **Calc tools**: margin/growth/CAGR functions verified against hand-computed values; the
  safe expression evaluator was tested against real Python sandbox-escape patterns
  (`__import__`, `().__class__.__bases__`, lambdas, comprehensions) — all rejected.
- **Agent graph**: a genuine multi-step plan (SQL -> Calc, with Calc referencing SQL's row
  output by index) executes correctly against the live DB. A failing step correctly halts
  the plan before a dependent step runs with garbage inputs. Fabricated citations are
  dropped; real ones are kept.
- **Evals**: the judge harness was proven to receive the *actual* query/answer/step-data
  (not a stub) by asserting on prompt contents in tests.
- **API**: FastAPI `/query` endpoint tested end-to-end via `TestClient` against the live DB,
  including the malformed-plan error path, rate limiting, LLM-failure handling, and a check
  that importing the app never requires an API key or loads a model.

**Not verified in the build sandbox** (no access to Hugging Face, Groq, Docker or Render there):
the real embedding model download, a live Groq call, `docker build`, and the Render deploy itself.
The LLM client is tested against a local server that mimics the OpenAI chat-completions shape,
and the real `start.sh` was run end to end against Postgres with that stand-in LLM.
Your first Render deploy is the first time the remaining pieces run for real.

## Run locally

```bash
cp .env.example .env            # then paste a free key from https://console.groq.com/keys
docker compose up --build       # Postgres + app
# open http://localhost:10000
```

Without Docker: `pip install -r requirements-dev.txt`, point `DATABASE_URL` (or the `DB_*`
variables) at a Postgres, run `python -m db.init_db`, then
`uvicorn api.main:app --port 10000`. Tests: `pytest calc/ retrieval/ sql_node/ agent/ evals/ api/`
(DB-dependent tests auto-skip when Postgres is unreachable).

## Deploy on Render

1. Push this folder to a GitHub repo.
2. Get a free API key at <https://console.groq.com/keys> (no card).
3. In Render: **New -> Blueprint**, select the repo. Render reads `render.yaml` and creates a
   Docker web service and a Postgres database.
4. When prompted, paste the key as `LLM_API_KEY`. Apply.
5. First build takes several minutes (it downloads the embedding model into the image).
   Then open the service URL. Check `/health`, and `/api/status` should show `llm_configured: true`.

On every start, `start.sh` creates the tables and loads the sample invoices if the table is empty.

**Free tier behaviour to expect** (verify on Render's pricing page, limits change):
- The web service sleeps after 15 minutes without traffic and takes about a minute to wake.
- Free Postgres expires 30 days after creation, with a 14-day grace period to upgrade before the
  data is deleted. Plan to recreate it (the app reseeds itself) or move to a paid database.
- 512 MB RAM. The default setup (ONNX embeddings, no reranker, one worker) is sized for that.
- Groq's free tier is rate limited, so the app retries on 429 and also limits each IP to
  8 questions per minute (`RATE_LIMIT_PER_MINUTE`) so a public URL can't drain your quota.

## Switch LLM provider

Three environment variables, no code change. Anything that speaks OpenAI `/chat/completions`:

| Provider | `LLM_BASE_URL` | Example `LLM_MODEL` |
|---|---|---|
| Groq (default) | `https://api.groq.com/openai/v1` | `llama-3.3-70b-versatile` |
| OpenRouter | `https://openrouter.ai/api/v1` | `meta-llama/llama-3.3-70b-instruct:free` |
| Together, Cerebras, a self-hosted vLLM/Ollama | their `/v1` URL | their model id |

The planner, SQL generator and synthesizer all need strict JSON or SQL output, so prefer a 70B-class
model. Small models fail the JSON parsing more often.

## Use your own data

- **Invoices:** `python -m ingestion.load_invoices your_invoices.csv` (same columns as
  `ingestion/sample_data/invoices.csv`) against the deployed database, or replace the sample CSV
  before deploying. A Render free database only accepts internal connections (`ipAllowList: []`),
  so for remote loading either use the Render shell or temporarily allow your IP.
- **Filings:** parsing PDFs needs Docling and PyTorch, which a 512 MB instance can't run. Do it
  once locally, commit the result:

  ```bash
  pip install -r requirements-ingest.txt
  python -m retrieval.export_corpus path/to/filings/ --company "Acme Corp" --fiscal-period FY2024 \
      --out data/corpus.jsonl --embed
  ```

  `--embed` precomputes vectors so the server skips embedding at startup. The bundled
  `data/corpus.jsonl` is one synthetic 10-K excerpt, for demo only. Set `ENABLE_RERANKER=true`
  if you have headroom (about 100 MB more RAM).

## Design decisions at a glance

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph, plan-then-execute | inspectable/debuggable plan vs. an opaque ReAct loop |
| Filing parsing | Docling (offline, local) | preserves table structure other parsers flatten |
| Chunking | Hybrid: structure-aware text + atomic tables | prevents splitting a table mid-row |
| Embeddings | BGE small via fastembed (ONNX) | same model family, no PyTorch, fits 512 MB |
| Vector DB | Qdrant in-process, rebuilt from `corpus.jsonl` at startup | no extra server; corpus is small |
| Retrieval | Hybrid (dense+BM25, RRF), optional reranker | exact terms (tickers, line items) + semantic match |
| Text-to-SQL | LLM generation + sqlglot validation + read-only transaction (+ read-only role when available) | defense in depth, works on hosts that block role creation |
| Calc | Deterministic dispatch + LLM tool-calling for edge cases | never let the LLM re-type a number |
| Grounding | Structured citations, validated against real chunk ids | fabricated citations get dropped, not trusted |
| Evals | LLM-as-judge (retrieval/SQL/end-to-end) | scales past a hand-labeled set; calibrate before trusting at scale |
| LLM provider | Open model behind an OpenAI-compatible API (Groq free tier) | zero cost, swap provider with env vars |
| Deployment | Render: Docker web service + Postgres via Blueprint | one-click from a repo |
