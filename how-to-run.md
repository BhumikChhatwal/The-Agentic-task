# How to run Byte Analyst

Run from the repository root (`C:\Users\bhumi\OneDrive\Documents\Onedrive\Desktop\Imps\codes\Projects\Byte`).

## 0. Check Python

Byte Analyst needs Python 3.10+. The dev setup is inside `.venv`. If you cloned
fresh, recreate it once:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows (cmd/PowerShell): .venv\Scripts\activate
pip install -e .
```

`pip install -e .` installs the package so `byteanalyst` is on `PATH` and the
editable flag keeps tests/imported code in sync with the source.

If you prefer to install all extras (web UI, charting deps) in one shot:

```bash
pip install -e ".[web,tests]"
```

## 1. Pick a mode

### No API key — offline demo (the one I kept running)

```bash
byteanalyst demo
byteanalyst demo --dataset employees
byteanalyst demo --dataset weather
```

`demo` walks the bundled datasets with the deterministic rule-based planner.
No key, no network, no model. The offline planner issues the same tool calls and
composes answers from tool output only — that is what lets this run offline.

```bash
byteanalyst ask "total revenue by region" --data data/ecommerce_orders.csv
byteanalyst ask "how does temp_c trend over time by city?" --data data/weather_daily.csv
byteanalyst ask "are salary differences between departments significant?" --data data/employees.csv
```

The offline planner is used automatically when no provider key is present.

### Pure profiling — no model, no agent

```bash
byteanalyst profile data/support_tickets.csv
byteanalyst profile data/weather_daily.csv --json
```

`profile` runs `profile_dataset` directly and prints a table + detail keys. It
does not talk to any model; it is the "what is in this file" command.

### With a real model

Environment variables beat CLI flags for secrets. The syntax differs by shell — use the one matching your terminal.

**Windows PowerShell (most likely on this machine):**

```powershell
$env:BYTEANALYST_API_KEY="sk-..."
$env:BYTEANALYST_MODEL="gpt-4o-mini"    # optional

byteanalyst ask "which region is growing fastest and is the gap significant?" `
  --data data/ecommerce_orders.csv --data data/customers.csv
```

**Windows cmd:**

```cmd
set BYTEANALYST_API_KEY=sk-...
set BYTEANALYST_MODEL=gpt-4o-mini

byteanalyst ask "which region is growing fastest and is the gap significant?" ^
  --data data/ecommerce_orders.csv --data data/customers.csv
```

**bash / zsh / Git Bash / WSL:**

```bash
export BYTEANALYST_API_KEY=sk-...
export BYTEANALYST_MODEL=gpt-4o-mini

byteanalyst ask "which region is growing fastest and is the gap significant?" \
  --data data/ecommerce_orders.csv --data data/customers.csv
```

#### Local model server (no API key, but the server must be running)

Any OpenAI-compatible server works: Ollama, vLLM, LM Studio, llama.cpp server.
A `WinError 10061 … actively refused` error means nothing is listening on that
port — the server is not running (or not installed).

**One command (recommended):**

```powershell
# Windows
.\scripts\setup_local_model.ps1
```

```bash
# Git Bash / WSL / macOS / Linux
./scripts/setup_local_model.sh
```

The script installs Ollama (if missing), starts the server, pulls the model,
writes `.env` so every future session needs no flags, and verifies the whole
chain end to end. Options: `-Model llama3.1` / `MODEL=llama3.1` for a
different model; `-SkipInstall` / `SKIP_INSTALL=1` if Ollama is already present.

The default model is **qwen2.5:7b** — measured to drive this agent's tool loop
far more reliably than llama3.1 on a CPU-only machine (llama3.1 8B frequently
ignores the tool schemas and answers from memory).

**Manual setup on Windows:**

1. Install: `winget install Ollama.Ollama`, or download `OllamaSetup.exe` from
   <https://ollama.com/download/windows> (per-user, no admin needed).
2. Pull a model (a few GB the first time): `ollama pull qwen2.5:7b`
3. Confirm it works: `ollama list`. The Windows app keeps the server running in
   the background; with the standalone CLI, start it yourself: `ollama serve`.

**Every session (PowerShell):**

```powershell
$env:BYTEANALYST_BASE_URL="http://localhost:11434/v1"
$env:BYTEANALYST_MODEL="qwen2.5:7b"
byteanalyst ask "profile the data" --data data/weather_daily.csv
```

**bash / Git Bash / WSL:**

```bash
export BYTEANALYST_BASE_URL=http://localhost:11434/v1
export BYTEANALYST_MODEL=qwen2.5:7b
byteanalyst ask "profile the data" --data data/weather_daily.csv
```

Setting `BYTEANALYST_BASE_URL` alone selects the local server — no key and no
`BYTEANALYST_PROVIDER` needed. Use an exact model name from `ollama list` (tags
included, e.g. `qwen2.5:7b`); a wrong name returns HTTP 404. If you just want
it to work with no server at all, add `--provider offline`.

Two knobs matter for local models on CPU (the setup script sets both):

- `BYTEANALYST_TIMEOUT=600` — loading a multi-GB model into RAM can exceed the
  120 s default on first use.
- `BYTEANALYST_TOOLS=profile_dataset,run_sql,run_python,statistical_test,detect_anomalies`
  — a scoped toolbox. 7-8B models call tools far more reliably with ~5 tools
  than with the full 10; the three memory tools, which small models fixate on,
  are the ones left out. Delete the line to get the full set.

Environment variables (preferred over CLI flags for secrets):

| Variable | Purpose |
|---|---|
| `BYTEANALYST_API_KEY` | Model key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `BYTEANALYST_PROVIDER` | `openai_compatible` \| `anthropic` \| `offline` (auto-detected if unset) |
| `BYTEANALYST_BASE_URL` | Base URL for any OpenAI-compatible server |
| `BYTEANALYST_MODEL` | Model name |
| `BYTEANALYST_TOOLS` | Comma-separated tool allowlist; unset means the full set |
| `BYTEANALYST_MAX_ITERATIONS` | Tool-call budget per question (default 12) |
| `BYTEANALYST_STATE_DIR` | Where memory, artifacts and transcripts live (default `./.byteanalyst`) |
| `BYTEANALYST_MEMORY` / `BYTEANALYST_AUTO_MEMORY` | Turn memory off entirely / only turn off automatic writes |
| `BYTEANALYST_TEMPERATURE` | Sampling temperature (default 0.0) |
| `BYTEANALYST_TIMEOUT` | Per-call timeout in seconds (default 120) |
| `BYTEANALYST_MAX_TOOL_OUTPUT_CHARS` | Tool output truncate point (default 6000) |

A `.env` at the repo root is loaded automatically; CLI flags win over it.

### Anthropic

```bash
set BYTEANALYST_PROVIDER=anthropic
set ANTHROPIC_API_KEY=sk-ant-...
byteanalyst ask "profile the data" --data data/employees.csv
```

### Interactive chat

```bash
byteanalyst chat --data data/ecommerce_orders.csv --data data/customers.csv
# inside: type a question, /tools /memory /reset /quit, or Ctrl-D
```

### Web UI

```bash
byteanalyst serve --data data/ecommerce_orders.csv --data data/customers.csv
# opens http://127.0.0.1:8000
```

The web UI is the same pipeline as the CLI: trace, grounding verdict, chart
artifacts and memory panel all render there.

## 2. Flags worth knowing

| Flag | Meaning |
|---|---|
| `--data FILE` (repeatable) | One or more tables to load (CSV/TSV/JSON/JSONL/Parquet/Excel) |
| `--name N` (repeatable) | Override the table name for each `--data` (order matters) |
| `--trace` | Show the tool trace (default on; use it to override `--quiet`) |
| `--quiet` | Hide the tool trace |
| `--no-color` | Disable ANSI colours |
| `--json` | Machine-readable JSON on stdout; the human trace is suppressed |
| `--session-file PATH` | Write a full transcript JSON to disk on exit |
| `--state-dir DIR` | Override where `.byteanalyst/` lives for memory/artifacts/transcripts |
| `--no-memory` / `--no-auto-memory` | Disable persistent memory / only disable automatic writes |
| `--provider` / `--model` / `--base-url` / `--api-key` | Override env-based configuration |
| `--max-iterations N` | Tool-call budget per question |

`byteanalyst tools` lists the tool catalogue; `byteanalyst memory list|search|add|edit|forget|clear|export|import|stats|history` inspects the persistent store.

## 3. File formats it accepts

`.csv`, `.tsv`, `.txt`, `.psv`, `.json`, `.jsonl`, `.ndjson`, `.parquet`, `.xlsx`, `.xls`.
Delimiters are sniffed; encodings fall back UTF-8 → cp1252 → latin-1. Currency symbols,
Excel-style negatives `(1,234)`, and long all-digit strings are handled so leading zeros
and money-like text survive as text when that is the right call.

```bash
byteanalyst ask "total price by category" --data ~/my_export.csv --data ~/branches.json
```

## 4. Verify it works

```bash
pytest -q
```

That is 186 tests, no network, no API key, ~30s. It exercises datasets, all seven
tools, memory, the agent loop, both LLM wire formats, the CLI, and the web app.

## 5. Where output ends up

By default everything lives in `./.byteanalyst/` relative to the working dir:

- `memory.db` — persistent memory SQLite store
- `artifacts/` — chart PNGs and other tool outputs
- transcript JSON when `--session-file` is used

Pass `--state-dir DIR` to move them elsewhere.

## 6. The two modes, clearly

- **LLM path (default with a key):** model chooses tools, reads results back,
  verifies every number against tool output, and writes durable memories for
  definitions/filters/corrections.
- **Offline path (no key):** a deterministic rule-based planner does the same
  tool-calling and verification. It is not a language model; it handles the
  common phrasings well and says plainly when it cannot map a question. It exists
  for keyless runs and for the automated tests.

Both paths share the verifier, both print `grounded: yes` / `unverified: ...` on
the answer footer.
