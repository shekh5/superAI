# SuperAI: Grounded Agent + Full CI/CD Pipeline

A FastAPI agent service with direct tool execution, a Gemini-powered ReAct loop,
Google Search-grounded current answers, session-persistent document retrieval,
Redis-backed conversations and traces,
browser interfaces, and a complete
Git → Docker → GitHub Actions → EC2 delivery pipeline.

This project demonstrates both agent orchestration and the infrastructure needed
to observe, test, package, and deploy it.

## Project structure

```
agentic-capstone/
├── app/
│   ├── main.py              # FastAPI app + direct tools + UI routes
│   └── static/
│       ├── chat.html         # Interactive chat interface
│       └── dashboard.html    # Observability dashboard
├── reasoning_chain/
│   ├── chain.py              # ReAct loop orchestrator (Gemini)
│   ├── context_compression.py # Loss-aware model-context compression
│   ├── decisions.py          # Strict agent-decision and tool-input validation
│   ├── documents.py          # Multi-format extraction, persistence, and retrieval
│   ├── prompts.py            # Versioned XML prompts + few-shot examples
│   ├── router.py             # /chain API routes + Redis persistence
│   ├── schemas.py            # Pydantic data contracts
│   └── tools.py              # Local, weather, and grounded web-search tools
├── tests/
│   ├── test_app.py           # API endpoint tests
│   ├── test_chain.py         # Reasoning chain logic tests
│   ├── test_documents.py     # Multi-format extraction and retrieval tests
│   ├── test_prompts.py       # Prompt contracts and XML boundary tests
│   └── test_self_correction.py # Bounded correction and recovery tests
├── docs/
│   └── screenshots/          # UI and API screenshots
├── .github/workflows/
│   ├── ci.yml                # lint + test + docker build check
│   └── cd.yml                # build, push to GHCR, deploy, rollback
├── Dockerfile                 # multi-stage, non-root, healthcheck
├── docker-compose.yml          # local dev: app + redis
├── docker-compose.prod.yml     # production: GHCR image + redis
├── requirements.txt
├── requirements-dev.txt
└── pyproject.toml             # ruff + pytest config
```

## Screenshots

### Chat Interface (`/chat`)
The interactive chat UI supports two orchestration modes: **Direct Agent** (manual
tool selection) and **Reasoning Chain** (goal-based multi-step reasoning via Gemini).

<p align="center">
  <img src="docs/screenshots/chat-ui.jpg" alt="Chat UI — Dark glassmorphism interface with sidebar, mode selector, and chat threads" width="800"/>
</p>

### Reasoning Chain in Action
A multi-step conversation showing the ReAct loop: the agent decomposes a goal into
tool calls, executes them, and returns a verified answer with a collapsible
execution trace timeline.

<p align="center">
  <img src="docs/screenshots/chat-conversation.jpg" alt="Chat showing weather query with execution trace — weather tool then calculator" width="800"/>
</p>

### Observability Dashboard (`/dashboard`)
Real-time metrics (success rate, latency, token usage), a searchable trace history
table, and a detailed trace inspector panel for debugging agent behavior.

<p align="center">
  <img src="docs/screenshots/dashboard-overview.jpg" alt="Dashboard with metric cards, trace history table, and inspector panel" width="800"/>
</p>

### Trace Inspector — Execution Path
Drill into any trace to see the full execution timeline: plan decomposition, each
tool step with input/output/latency/tokens, API call details, and verification
outcome.

<p align="center">
  <img src="docs/screenshots/dashboard-trace-inspector.jpg" alt="Trace inspector showing step-by-step execution path with weather and calculator tools" width="800"/>
</p>

### API Documentation (`/docs`)
Auto-generated Swagger UI showing all 14 endpoints across the core service and
reasoning chain modules.

<p align="center">
  <img src="docs/screenshots/swagger-api-docs.jpg" alt="FastAPI Swagger UI showing all endpoints grouped by default and chain" width="800"/>
</p>

### API Endpoints
Live JSON responses from the health check and root service info endpoints.

<p align="center">
  <img src="docs/screenshots/api-endpoints.jpg" alt="Health and root endpoint JSON responses" width="800"/>
</p>

## Run it locally (no Docker)

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
export GEMINI_API_KEY=your-key       # required for /chain routes
export WEB_SEARCH_MODEL=gemini-3.1-flash-lite # optional; defaults to GEMINI_MODEL
export REACT_TEMPERATURE=0.1         # optional server default, range 0.0-1.0
export SUMMARY_TEMPERATURE=0.2       # internal rolling-summary setting
export WEATHER_API_KEY=your-key      # optional; otherwise deterministic mock weather is used
uvicorn app.main:app --reload
```

Visit http://localhost:8000/docs for the interactive Swagger UI.

Try it:
```bash
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"tool": "calculator", "argument": "2 + 2 * 10"}'
```

## Run it with Docker

```bash
docker build -t agentic-capstone .
docker run -p 8000:8000 agentic-capstone
```

## Run the full stack (app + redis) with Compose

```bash
docker compose up --build
```

Redis uses append-only persistence on the `redis_data` named volume, so sessions
and traces survive ordinary container recreation. The chat UI also caches the latest
50 messages per session in browser storage as a fallback. `docker compose down -v`
intentionally deletes the Redis volume.

### Conversation context management

The model does not receive the full UI history or execution traces. Each session keeps
three separate Redis records: capped display messages, clean recent `user`/`model`
messages, and a rolling summary of older turns. The ReAct loop sends this memory with
Gemini's native roles and removes the oldest context when the configured input budget is
reached. System/tool instructions, the current goal, output reserve, and a safety margin
are accounted for independently. See [docs/context-management.md](docs/context-management.md)
for the key layout, defaults, and EC2 configuration guidance.

Context selection is priority-aware and adaptive: the latest four messages are high priority,
older recent turns are medium priority, and rolling memory is already compressed. At 60%, 80%,
and 90% planned utilization, increasingly strict compression levels apply. Oversized tool output
is shortened only in the model-facing payload while the full value remains in the trace. Context
usage, retained/dropped messages, and compression level are visible in each model call.

### Current web-grounded answers

The reasoning chain can select `web_search` for recent, changing, or explicitly web-sourced
questions. It uses Gemini's native Google Search grounding, so the existing `GEMINI_API_KEY`
is sufficient; no second search-provider secret is required. Successful search results must
contain public sources, and the final model answer must preserve at least one exact returned URL.
The chat safely renders those Markdown links, while the dashboard lists source and usage telemetry.

Grounded search calls may be billable. The tool therefore does not automatically retry a failed
search; a corrected query must be selected explicitly by the agent. Configure a separate supported
model with `WEB_SEARCH_MODEL` if needed. See
[docs/web-search-grounding.md](docs/web-search-grounding.md) for the flow and EC2 setup.

### Evidence Workspace document questions

In Reasoning Chain mode, the `Files` button accepts PDF, DOCX, TXT, Markdown, CSV, TSV, XLSX, and
PPTX documents. Legacy mode stores extracted chunks in Redis and uses local lexical retrieval.

Setting `EVIDENCE_WORKSPACE_ENABLED=true` enables the production workspace: invite-only accounts,
private S3 originals, asynchronous extraction, local multilingual embeddings, Neon/pgvector hybrid
retrieval, structured evidence, claim-support badges, and clickable PDF/text previews. Redis remains
responsible for clean conversation context, traces, caches, and the ingestion queue. See
[docs/evidence-workspace.md](docs/evidence-workspace.md) for provisioning and deployment, and
[docs/document-rag.md](docs/document-rag.md) for extraction limits.

Local workspace development uses the optional Compose profile:

```bash
docker compose --profile workspace up -d postgres minio minio-init redis
EVIDENCE_WORKSPACE_ENABLED=true AUTH_REQUIRED=true \
docker compose --profile workspace run --rm app alembic upgrade head
EVIDENCE_WORKSPACE_ENABLED=true AUTH_REQUIRED=true \
BOOTSTRAP_ADMIN_EMAIL=admin@example.com \
BOOTSTRAP_ADMIN_PASSWORD='replace-with-a-long-password' \
docker compose --profile workspace up --build
```

## Run tests

```bash
pytest -v
ruff check app reasoning_chain tests
```

## Setting up the pipeline on GitHub

1. Create a new repo on GitHub and push this code:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: agentic capstone scaffold"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **CI** (`ci.yml`) runs automatically on every push/PR — no setup needed.
   It lints with `ruff`, runs `pytest` across Python 3.11 and 3.12, and
   verifies the Docker image builds.

3. **CD** (`cd.yml`) runs on every push to `main`. It:
   - Builds and pushes your image to GitHub Container Registry (GHCR)
   - Tags it with both the git SHA (traceable) and `latest`
   - Runs a placeholder deploy step
   - Health-checks the deployed service
   - Rolls back automatically if the health check fails

   To deploy to EC2, add these **repo secrets**
   (Settings → Secrets and variables → Actions):
   - Existing: `EC2_HOST`, `EC2_USER`, `EC2_SSH_KEY`, `GHCR_PAT`, `GEMINI_API_KEY`
   - Evidence Workspace: `EVIDENCE_WORKSPACE_ENABLED`, `AUTH_REQUIRED`, `DATABASE_URL`,
     `S3_BUCKET`, `AWS_REGION`, `COOKIE_SECURE`, `BOOTSTRAP_ADMIN_EMAIL`, and
     `BOOTSTRAP_ADMIN_PASSWORD`

## Suggested learning path through this repo

1. Get it running locally, understand `main.py`
2. Make a change on a feature branch, open a PR, watch CI run
3. Intentionally break a test — watch CI fail and block the merge
4. Merge to `main` — watch CD build+push an image to GHCR
5. Pick a free host (Fly.io or Render) and wire up the real deploy step
6. Break the `/health` endpoint on purpose, deploy, and watch the
   rollback step trigger
7. Add a 4th tool to the agent and repeat the whole cycle

## Where this goes next (once comfortable)

- Swap the mock tools for real ones (weather API, real DB-backed memory)
- Consolidate the direct and reasoning-chain tool registries
- Add authentication, rate limiting, and access controls for traces
- Add a staging environment + manual approval gate before prod deploy


# Reasoning chain

The active orchestrator uses a bounded ReAct loop: Gemini proposes one validated
tool action, the service executes it, and the updated history is returned to Gemini
until the goal is satisfied or the eight-step limit is reached.

The model stages use centralized, versioned XML system prompts and four compact
few-shot examples. Dynamic goals, summaries, and tool results remain untrusted model
content and are XML-escaped. ReAct outputs include a brief action-selection `reason`
instead of requesting detailed hidden chain-of-thought; legacy `thought` responses are
still accepted. See [docs/prompt-engineering.md](docs/prompt-engineering.md).

ReAct decisions are validated with typed action/final schemas and per-tool input contracts.
Malformed JSON, unknown tools, invalid arguments, duplicate failed actions, and unsupported
success claims receive at most two model-output correction attempts. Corrections do not consume
tool steps, and exhaustion returns a controlled unsatisfied trace. The runtime tool registry has
no filesystem, shell, Git, or code-editing capability. See
[docs/agent-self-correction.md](docs/agent-self-correction.md).

## Wire it in

```python
# main.py (or wherever your FastAPI app is created)
from reasoning_chain.router import router as chain_router
app.include_router(chain_router, prefix="/chain")
```

The tool layer uses a bounded arithmetic parser, IANA timezone handling, optional
WeatherAPI.com integration, Google Search grounding, retries, and a circuit breaker. Failure
injection is off by default and can be enabled with `WEATHER_FAILURE_RATE` or
`CALCULATOR_BAD_INPUT_RATE`, using values between `0` and `1`.

## Endpoints

- `POST /chain/plan?goal=...` — decomposition only, no tools run. Use this
  first to sanity-check the model's reasoning.
- `POST /chain/run?goal=...&session_id=...` — run the ReAct loop and return its trace.
  An optional `temperature=0.0..1.0` query parameter overrides the server default for that run.
- `GET /chain/traces` — list recent trace summaries.
- `GET /chain/trace/{request_id}` — replay a past run from Redis.
- `/chain/session/...` routes — save session metadata and conversation history.
- `POST/GET /chain/session/{session_id}/documents` — upload or list session documents.
- `DELETE /chain/session/{session_id}/documents/{document_id}` — delete extracted content.

## Try it locally

```bash
export GEMINI_API_KEY=...
export WEB_SEARCH_MODEL=gemini-3.1-flash-lite  # optional
uvicorn app.main:app --reload
curl -X POST "http://localhost:8000/chain/plan?goal=what+time+is+it+and+is+it+raining+in+Tokyo"
curl -X POST "http://localhost:8000/chain/run?goal=what+time+is+it+and+is+it+raining+in+Tokyo"
curl -X POST "http://localhost:8000/chain/run?goal=explain+the+weather&temperature=0.4"
curl -X POST "http://localhost:8000/chain/run?goal=what+are+today%27s+top+AI+updates"
```

To demonstrate recovery behavior locally, set `WEATHER_FAILURE_RATE=0.35`.
Production deployments should leave failure injection unset or explicitly set to `0`.

## Tests

```bash
pip install pytest --break-system-packages
pytest tests/test_chain.py -v
```

All LLM calls are mocked, so this runs in your existing CI (Python
3.11/3.12 matrix) without needing an API key or network access.

## What to look at once it's running

1. **`/chain/plan`** — read the JSON. Does the model's decomposition make
   sense for a goal you didn't anticipate? This is where you'll spend most
   of your debugging time in real agent work.
2. **Force a failure** — run with `WEATHER_FAILURE_RATE=1.0` and hit
   `/chain/run`. Confirm the circuit breaker kicks in
   and `final_summary` is honest about what's missing, instead of the
   model quietly making up a temperature.
3. **Model-call telemetry** — inspect prompts, token usage, tool inputs, outputs,
   retries, and API latency in `/dashboard`.

## Next step (Phase 2 memory)

Session context and trace history are stored in Redis. A next memory phase would
add explicit retention limits, summarization, and user-level isolation.
