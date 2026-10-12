# Byte Analyst

A data analysis **agent** that answers natural-language questions about
arbitrary tabular data by calling its own tools — and by refusing to answer from
memory.

Ask it things like:

```
byteanalyst ask "total revenue by region" --data data/ecommerce_orders.csv
byteanalyst ask "are there outliers in revenue, and which category are they in?" --data data/ecommerce_orders.csv
byteanalyst ask "are salary differences between departments real or noise?" --data data/employees.csv
byteanalyst ask "how does temp_c trend over time by city?" --data data/weather_daily.csv
```

Every number in an answer is traced back to a tool result. If the model states a
figure that no tool produced, the agent catches it and sends the model back to
fix it before the user ever sees it.

---

## Contents

- [What I built (and what is mine)](#what-i-built-and-what-is-mine)
- [Quickstart](#quickstart)
- [The seven analysis tools](#the-seven-analysis-tools)
- [Persistent memory (deep-dive mission)](#persistent-memory-deep-dive-mission)
- [How hallucination is prevented](#how-hallucination-is-prevented)
- [Architecture](#architecture)
- [Demos](#demos)
- [Bring your own data](#bring-your-own-data)
- [Bundled datasets](#bundled-datasets)
- [Tests](#tests)
- [Limitations and what I would do next](#limitations-and-what-i-would-do-next)

---

## What I built (and what is mine)

Everything in this repository was written for this task. No agent framework
(LangChain, LlamaIndex, CrewAI, AutoGen, …) is used — the tool-calling loop, the
memory store, the retrieval scorer and all seven analysis tools are implemented
here, on top of general-purpose libraries (pandas, numpy, scipy, statsmodels,
scikit-learn, DuckDB, matplotlib, httpx, FastAPI).

Honest breakdown of the external inputs:

| Component | What I used | What is mine |
|---|---|---|
| Language model | A hosted or local **OpenAI-compatible** endpoint, or Anthropic. Configured by env var; **no vendor is hard-coded**. | The provider adapter, the tool schemas, the system prompt, the loop, the verification and the repair turn. |
| Tool-calling protocol | Standard function-calling JSON schema | A translator for two wire formats, plus recovery for models that emit near-JSON arguments. |
| Statistics & numerics | `scipy`, `statsmodels`, `scikit-learn`, `pandas`, `numpy` | The test selection, effect sizes, confidence intervals, plain-language interpretation, and a reliability check that flags results you should not trust. |
| SQL engine | `duckdb` (in-process) | Schema exposure, read-only guard, cross-table joins, result summarisation, and the schema hint attached to every SQL failure. |
| Charts | `matplotlib` | Chart construction, the numeric summary *of what was plotted* that the agent must reason from, and artifact handling. |
| Embeddings | Optional and off by default | A pluggable hook; retrieval is a hybrid lexical + context-key scorer so the default path needs no extra service. |

No part of the system is hard-coded for one dataset. The schema, the column roles,
the prompt and the tool arguments are all derived at runtime from whatever file you
load. To check that claim rather than assert it, I authored two files *after* the
code was finished — a hand-written TSV with currency-symbol prices and a nested
JSON table — and ran the agent against them, alongside the five bundled datasets
that the code also special-cases nowhere (see
[Bring your own data](#bring-your-own-data)).

**Design decision worth calling out:** the repository runs end-to-end with **no API
key at all**. *No key at all* means two different things depending on what you want:

- **Offline demo / tests — no model at all.** When no LLM key is present, the agent
  swaps the model for a deterministic rule-based planner (`byteanalyst/llm/offline.py`)
  that issues the same tool calls and composes answers *from tool output only*. That is
  what makes the whole pipeline testable and demoable offline — and the answer footer
  says so explicitly.
- **Local model server — no API key, but a server still runs.** You can point the agent
  at a local OpenAI-compatible server (Ollama, vLLM, LM Studio, llama.cpp serve, …)
  with `BYTEANALYST_BASE_URL=http://localhost:11434/v1`. No API key is needed, but the
  server itself must be running on the machine. The README's "Full agent run" section
  has the exact PowerShell and bash command lines.

The LLM path (real hosted model) is the default when a key is present.

---

## Quickstart

Requires Python 3.10+.

```bash
git clone <your-fork-url> byte-analyst && cd byte-analyst
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .                   # or: pip install -r requirements.txt
cp .env.example .env               # optional: add a model key
```

### Zero-config run (no API key)

```bash
byteanalyst demo                              # every bundled dataset, offline planner
byteanalyst ask "total revenue by region" --data data/ecommerce_orders.csv
byteanalyst profile data/support_tickets.csv  # pure profiling, no model at all
byteanalyst serve                             # web UI at http://127.0.0.1:8000
```

### Full agent run (with a model)

Environment variables beat CLI flags for secrets. The syntax differs by shell —
use the one matching your terminal. The examples below are bash; for PowerShell use
`$env:NAME="value"` and for cmd use `set NAME=value`.

```bash
# OpenAI (bash / Git Bash / WSL)
export BYTEANALYST_API_KEY=sk-...            # or put it in .env
export BYTEANALYST_MODEL=gpt-4o-mini

byteanalyst ask "which region is growing fastest and is the gap significant?" \
  --data data/ecommerce_orders.csv --data data/customers.csv --trace
```

**PowerShell equivalent:**

```powershell
$env:BYTEANALYST_API_KEY="sk-..."
$env:BYTEANALYST_MODEL="gpt-4o-mini"
byteanalyst ask "which region is growing fastest and is the gap significant?" `
  --data data/ecommerce_orders.csv --data data/customers.csv --trace
```

Any OpenAI-compatible endpoint works — OpenRouter, Groq, Together, vLLM, LM Studio,
Ollama:

```bash
# local Ollama, no key needed (bash / Git Bash / WSL)
export BYTEANALYST_BASE_URL=http://localhost:11434/v1
export BYTEANALYST_MODEL=llama3.1
byteanalyst ask "profile the data" --data data/weather_daily.csv
```

**PowerShell equivalent:**

```powershell
$env:BYTEANALYST_BASE_URL="http://localhost:11434/v1"
$env:BYTEANALYST_MODEL="llama3.1"
byteanalyst ask "profile the data" --data data/weather_daily.csv
```

When `BYTEANALYST_BASE_URL` is set, the agent sends OpenAI-format requests to that
server — Ollama, vLLM, LM Studio, llama.cpp serve, etc. — so no API key is needed
for a local model.

Anthropic is supported natively. Environment variables: `BYTEANALYST_PROVIDER=anthropic`
and `ANTHROPIC_API_KEY=sk-ant-...` (PowerShell: `$env:ANTHROPIC_API_KEY="..."`).

| Environment variable | Purpose |
|---|---|
| `BYTEANALYST_PROVIDER` | `openai_compatible` \| `anthropic` \| `offline` (auto-detected from the keys if unset) |
| `BYTEANALYST_API_KEY` | Model API key (falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) |
| `BYTEANALYST_BASE_URL` | Base URL for any OpenAI-compatible server |
| `BYTEANALYST_MODEL` | Model name |
| `BYTEANALYST_MAX_ITERATIONS` | Tool-call budget per question (default 12) |
| `BYTEANALYST_STATE_DIR` | Where memory, artifacts and transcripts live (default `./.byteanalyst`) |
| `BYTEANALYST_MEMORY` / `BYTEANALYST_AUTO_MEMORY` | Turn memory off entirely / only turn off automatic writes |

### Command reference

```
byteanalyst ask "<question>" [--data FILE]... [--trace|--quiet] [--json] [--session-file F]
byteanalyst chat   [--data FILE]...            # multi-turn; /tools /memory /reset /quit
byteanalyst profile FILE [--columns A B] [--json]
byteanalyst memory list|search|add|edit|forget|clear|export|import|stats|history
byteanalyst demo   [--dataset orders|employees|weather|tickets|all]
byteanalyst tools  [--json]
byteanalyst serve  [--host H] [--port P] [--data FILE]...
```

---

## The seven analysis tools

The requirement was five *genuinely different* tools — not five wrappers over the
same operation. These seven do different kinds of work, and the agent picks by
what the question needs.

| # | Tool | Kind of analysis | Why it is not a wrapper |
|---|---|---|---|
| 1 | `profile_dataset` | Understanding | Computes no relationship at all: it characterises the table itself — inferred semantic role per column, nulls, cardinality, ranges, duplicates, quality warnings, and suggested follow-ups. It is what the agent runs first on unfamiliar data. |
| 2 | `run_sql` | Relational querying | Operates on relations, not columns: grouping, window functions, CTEs and **joins across several loaded tables** via DuckDB. Answers "what is the number". |
| 3 | `run_python` | General computation | Everything SQL cannot express in one statement — reshapes, cleaning chains, custom metrics, resampling loops. Runs in a **separate process** so a timeout actually kills it, with restricted builtins and an import allow-list. |
| 4 | `statistical_test` | Inference | Produces uncertainty, not a point estimate: 13 tests (t / Welch / paired / Mann-Whitney / Wilcoxon / ANOVA / Kruskal / chi-square / Pearson / Spearman / Kendall / Shapiro-Wilk / two-proportion) with effect sizes, confidence intervals, a plain-language conclusion gated on α — and explicit caveats when a result is not trustworthy. |
| 5 | `detect_anomalies` | Detection | Finds the *exceptions* rather than the summary. Five mechanisms with different assumptions: Tukey IQR fences, z-score, modified z-score on MAD, multivariate Isolation Forest, and seasonal-residual detection after decomposition. Optionally applied **within groups** so a regional level shift cannot mask a local outlier. |
| 6 | `analyze_time_series` | Temporal | Treats row order as information: resampling to a regular frequency, trend slope with R², STL seasonality strength, Augmented Dickey-Fuller stationarity, change-point detection, rolling averages, and forecasting (Holt-Winters ETS / ARIMA / naive / linear) with confidence intervals. |
| 7 | `visualize` | Presentation | Output is a picture — and, critically, it also returns a *numeric summary of exactly what was plotted*, so the agent's prose stays grounded in the same numbers that went into the figure. Ten chart types. |

Plus three tools that act on memory rather than data:
`remember`, `recall`, `manage_memory` (list / edit / forget / clear).

### How the agent chooses

The system prompt (`byteanalyst/agent/prompt.py`) sets the discipline: profile
before analysing, never state an unsupported number, cite the tool behind each
claim, and stop as soon as the question is answerable. Two rules do most of the
work in practice:

- **Look before you leap.** Guessing a column name wastes a turn, so the agent
  profiles first on unfamiliar schemas.
- **One tool call answers one question.** Multi-step questions chain calls
  (profile → query → test), and the loop ends when the model answers instead of
  calling a tool.

Tool errors are returned as readable results, not exceptions, so the model can
recover: a bad SQL query comes back with the real table/column list attached, an
unknown column comes back with the closest matches, and a timed-out computation
tells the model to narrow the data first.

---

## Persistent memory (deep-dive mission)

The store lives in SQLite at `.byteanalyst/memory.db` and is deliberately **small
and curated**. It does not record chat messages, and it does not dump datasets.
It records *durable facts that should change future behaviour*, in six typed kinds:

| Kind | Meaning | Example |
|---|---|---|
| `definition` | What a column, metric or term means | "`revenue` is net of refunds" |
| `preference` | How results should be presented | "weekly summaries, not daily" |
| `correction` | An override of an earlier belief | "churn is measured weekly, not monthly" |
| `filter` | A reusable, named population | "active customers exclude cancelled accounts" |
| `fact` | A verified, durable fact about the data | "fiscal year starts in February" |
| `note` | Background context, lowest priority | — |

### What gets kept

Nothing is written unless it looks durable. The model is given an explicit policy
(*definitions, preferences, corrections, reusable filters — never one-off numbers
or anything derivable from the data*), and after each turn a separate extraction
pass asks the model to propose candidates, which are then parsed and validated
(kind whitelist, length bounds, key normalisation). Automatically-writable memory
can be switched off with `--no-auto-memory`.

### Retrieval — why "relevant" is not just keyword search

Pure lexical search misses the most important memory type. "The column `status`
means …" is relevant whenever a table containing `status` is loaded, even if the
question never says "status". So retrieval blends three signals
(`byteanalyst/memory/retrieval.py`):

1. **Lexical overlap** — IDF-weighted token overlap against the question, computed
   over the small in-memory corpus (a BM25-flavoured score).
2. **Context-key match** — overlap between the memory's declared `keys` and the
   tables/columns *actually loaded in this session*, plus a bonus when the question
   mentions a key verbatim.
3. **Authoritative-kind prior** — corrections and filters outrank notes, modulated
   by confidence, recency and past usefulness. `uses`/`last_used_at` are bumped each
   time a memory is surfaced, so memories that keep proving useful rank higher.

An **optional embedder hook** can blend in cosine similarity; the default path is
fully offline and deterministic. When it is enabled, semantic recall is computed
*before* the relevance gate, so a memory with zero lexical overlap ("churn" for a
query about "attrition") can still be retrieved.

### Conflicts, staleness and inspection

- Identical text is **reinforced** (confidence rises) — never duplicated.
- The same kind+keys with different text **supersedes**: the older record is kept as
  history with `active = 0` and `superseded_by` set. Nothing is silently lost.
- A `correction` overrides prior beliefs with overlapping subject or near-identical
  wording, whatever kind they were.
- Every write is also appended to an audit log (`memory` table + `memory_events`).
- Scoped memories (`scope = orders`) do not collide with the same key in another
  dataset.

Users can inspect and control all of it:

```bash
byteanalyst memory list                    # everything active
byteanalyst memory history                 # audit trail (created / reinforced / superseded / forgotten)
byteanalyst memory search "active customer" # ranked, with the reason each matched
byteanalyst memory forget <id>             # deactivate (kept in history)
byteanalyst memory clear --yes             # deactivate everything
byteanalyst memory export > memories.json  # full dump, re-importable
```

### Memory that actually changes behaviour

Remembered **filters and definitions are applied as SQL predicates**, and the agent
is instructed to say so. The clearest demonstration is in `byteanalyst demo --dataset orders`:

```
how many active customers are there by region
  → select "region", count(*) from ecommerce_orders group by 1          -- cancelled rows included

user: "remember that active customers exclude cancelled accounts"
  → memory created [752f36…] (filter) active customers exclude cancelled accounts

new agent, same memory — "how many active customers are there by region"
  → select "region", count(*) from ecommerce_orders
      where (status <> 'cancelled') group by 1                          -- 836 → 798 for North
```

A memory remembered for one table is **not** applied to another: the predicate is
only attached when every column it references exists in the current table, so a
rule about `invoice_status` cannot break a query over `orders`.

The same flow is available in the browser: the memory panel writes facts, and the
next question shows the recall in the trace.

---

## How hallucination is prevented

This is the part I care most about, because "the agent analyses data" is exactly
what a chat wrapper cannot do.

1. **The prompt forbids unsupported numbers** and requires a tool call before any
   quantitative claim.
2. **Every tool result is rendered back into the conversation** — including the
   table, so the model reasons over rows it actually received.
3. **Verification after every answer** (`find_unsupported_numbers`): every numeric
   literal in the prose is checked against the numbers the tools produced, allowing
   for rounding and ignoring years and list ordinals.
4. **One repair turn.** If anything is unsupported, the model is sent back with the
   exact list — *"these numbers were not produced by any tool call: 987654. Either
   call the tool that computes them, or remove them"* — and only then is an answer
   accepted. Anything still unsupported is reported to the user as unverified
   rather than hidden.

The CLI/web UI print the verdict (`grounded: yes` / `unverified: …`), and the same
check runs on the offline planner's output, so the guarantee is not specific to the
model path. `tests/test_agent.py::test_llm_loop_escalates_unverified_numbers_once`
exercises the whole escalation.

---

## Architecture

```
byteanalyst/
  datasets.py        arbitrary file loading (CSV/TSV/JSON/JSONL/Parquet/Excel),
                     delimiter + encoding sniffing, type coercion, semantic roles,
                     the registry, and DuckDB registration
  tools/
    base.py          Tool ABC, ToolContext, column resolution with suggestions
    registry.py      schemas for the model + safe dispatch (errors never escape)
    profiling.py     profile_dataset
    sql.py           run_sql        (read-only guard, schema hints on failure)
    python_exec.py   run_python     (subprocess + _exec_child.py sandbox)
    stats_tests.py   statistical_test
    anomalies.py     detect_anomalies
    timeseries.py    analyze_time_series
    visualize.py     visualize
    memory_tools.py  remember / recall / manage_memory
  memory/
    models.py        typed records, kinds, save outcomes
    store.py         SQLite persistence, conflict resolution, audit log, control
    retrieval.py     hybrid scorer (+ optional embedder)
  llm/
    base.py          provider-neutral messages + JSON recovery helpers
    openai_compat.py OpenAI / OpenRouter / Groq / vLLM / Ollama / LM Studio
    anthropic.py     native Messages API
    offline.py       deterministic rule-based planner (keyless mode)
    scripted.py      test double for the loop
  agent/
    prompt.py        the system prompt and memory-extraction prompt
    core.py          the loop: retrieve → act → verify → write back
  web/               FastAPI app + single-page UI
  cli.py, render.py  command line and terminal rendering
scripts/make_sample_data.py   seeded generator for the bundled datasets
tests/                        186 tests
```

Data flow for one question:

```
question
  → retrieve relevant memories (hybrid scorer) ── inject into system prompt
  → loop: model ↔ tool execution (result rendered back into the conversation)
  → verification: are all numbers supported by tool output?
        ├─ no  → one repair turn
        └─ yes → answer
  → extraction pass: anything durable? → write to memory
```

---

## Demos

### Terminal

```bash
byteanalyst demo                      # all four scenarios, offline planner
byteanalyst demo --dataset employees  # t-test + chi-square on HR data
byteanalyst chat --data data/ecommerce_orders.csv --data data/customers.csv
```

`demo` prints the full trace (tool calls with arguments, results, verification,
memory recalls/writes) followed by the answer, so the reasoning is auditable.

### Web UI

```bash
byteanalyst serve --data data/ecommerce_orders.csv --data data/customers.csv
# open http://127.0.0.1:8000
```

The page shows the dataset chips, the tool catalogue, the memory panel
(inspect/add/forget), and a transcript where every turn renders its tool calls,
results, grounding verdict and any chart the agent produced.

---

## Bring your own data

Nothing is dataset-specific. Point it at a file:

```bash
byteanalyst ask "which product category has the worst satisfaction and is it significant?" \
  --data ~/my_export.csv --data ~/customers.json
```

The two files I wrote specifically to test generalisation, after the code was
complete (neither is referenced anywhere in the codebase):

- a hand-written **TSV** — tab delimiters, `£`-prefixed prices (currency symbols
  coerced to numeric), a date column, an empty-string column, and one deliberate
  price-entry error;
- a **nested JSON** file (`{"branches": [...]}`) loaded as a second, joinable table.

Plus the five bundled `data/` CSVs, which span e-commerce, HR, weather and
support-ticket shapes (2,600 / 400 / 320 / 6,576 / 1,800 rows).

Two behaviours worth noting from that run: the agent grouped a Coffee-vs-Food price
comparison correctly by recognising the *values* in the question, and it attached a
caveat — *"`Food` has only 4 observations, and every value is identical, so the
effect size is undefined"* — instead of reporting a confident p-value.

Formats supported: `.csv`, `.tsv`, `.txt`, `.psv`, `.json`, `.jsonl`, `.ndjson`,
`.parquet`, `.xlsx`, `.xls`. Delimiters are sniffed; encodings fall back through
UTF-8 → cp1252 → latin-1. Excel-style currency and `(1,234)` negatives are coerced;
long all-digit strings are kept as text so leading zeros survive.

---

## Bundled datasets

Generated by `scripts/make_sample_data.py` (seeded, so results are reproducible):

| File | Rows | Shaped to exercise |
|---|---|---|
| `data/ecommerce_orders.csv` | 2,600 | trend + weekly seasonality + a two-week demand slump (change point), 18 planted data-entry outliers, missing values, joins |
| `data/customers.csv` | 400 | a dimension table with no measures — makes joins necessary |
| `data/employees.csv` | 320 | HR data where ANOVA, t-tests, chi-square and correlation all have something real to find |
| `data/weather_daily.csv` | 6,576 | three cities, strong annual cycle, and a planted heatwave that IQR misses but seasonal-residual detection finds |
| `data/support_tickets.csv` | 1,800 | a deliberately messy table: constant column, 42%-null column, free text, heavy-tailed durations |

---

## Tests

```bash
pytest -q          # 186 tests, ~30s, no network, no API key
```

Coverage by area:

| File | What it pins down |
|---|---|
| `test_datasets.py` | Every file format, delimiter/encoding fallback, currency→numeric, leading-zero ids, role inference, duplicate columns, registry lookup, DuckDB joins |
| `test_tools_query.py` | Profiling all vs. one table, SQL joins/windows/guards, the schema hint on failure, subprocess execution, sandbox denials, timeout kill |
| `test_tools_stats.py` | Each test family, effect sizes, α-sensitivity, all forecast methods, change points, and every error message a model could hit |
| `test_tools_viz.py` | All ten chart kinds produce real PNGs, and the summaries match the data |
| `test_memory.py` | Reinforce/supersede/correct, scope isolation, retrieval relevance and reasons, control operations, persistence across instances, the embedder hook |
| `test_agent.py` | Offline routing, the scripted LLM loop, grounding escalation, iteration budget, memory injection changing SQL, memory write-back, events, failure paths |
| `test_llm.py` | Both wire formats, near-JSON argument recovery, retries, config auto-detection, factory rules |
| `test_cli.py` | Every subcommand through `main()`, including `--json` purity and that the demo never touches the real memory DB |
| `test_web.py` | All endpoints, uploads, inline charts, and per-request memory toggling |

---

## Limitations and what I would do next

Being explicit about the edges:

- **The rule-based planner is not a language model.** It handles the common
  phrasings well and is honest when it cannot map a question, but it cannot follow
  multi-hop reasoning the way a model does. It exists for keyless runs and tests.
- **`run_python` is a convenience sandbox, not a security boundary.** It restricts
  builtins and imports and runs out-of-process with a real timeout, but the code
  runs on your machine with your privileges. It is meant for the agent's own
  analysis snippets.
- **Verification checks numbers, not reasoning.** It catches fabricated figures and
  exact-value mistakes; it cannot catch a statistically wrong-but-supported claim.
  That is what the caveats on effect sizes are for.
- **The offline planner applies memory filters itself**; on the LLM path the model
  applies recalled rules, which is powerful but means a contradictory memory can
  still be followed. The `correction` kind and `memory history` exist to make that
  recoverable.
- **No multi-user isolation.** Memory and artifacts are per workspace
  (`.byteanalyst/`), which matches a local analysis tool.

Natural next steps: a `pandas`-native result cache so repeated questions skip
recomputation; statistical-tool support for regression and mixed models; a
chart-reading pass that lets the model *see* the figure it produced; and
per-column provenance so an answer can name the source file and row range behind
each figure.

---

## Licence

MIT.
