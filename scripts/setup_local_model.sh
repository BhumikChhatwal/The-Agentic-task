#!/usr/bin/env bash
# One-command setup for running Byte Analyst with a local Ollama model.
#
#   ./scripts/setup_local_model.sh                # qwen2.5:7b
#   MODEL=llama3.1 ./scripts/setup_local_model.sh     # different model
#   SKIP_INSTALL=1 ./scripts/setup_local_model.sh     # Ollama already installed
#
# Installs Ollama (if missing), pulls the model, writes .env, verifies end-to-end.
set -euo pipefail

MODEL="${MODEL:-qwen2.5:7b}"
cd "$(dirname "$0")/.."

step() { printf '\n==> %s\n' "$*"; }
ok()   { printf '    OK  %s\n' "$*"; }
warn() { printf '    !!  %s\n' "$*" >&2; }

server_up() {
  curl -sf -m 2 http://localhost:11434/api/tags >/dev/null 2>&1
}

# --- 1. Install Ollama -------------------------------------------------------
if [ "${SKIP_INSTALL:-0}" = "1" ]; then
  step "Skipping install (SKIP_INSTALL=1)"
elif command -v ollama >/dev/null 2>&1; then
  step "Ollama already installed"
else
  step "Installing Ollama"
  if command -v winget >/dev/null 2>&1; then          # Git Bash on Windows
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
  elif command -v brew >/dev/null 2>&1; then          # macOS
    brew install ollama
  elif [ "$(uname -s)" = "Linux" ]; then              # Linux
    curl -fsSL https://ollama.com/install.sh | sh
  else
    warn "No supported installer found. Install manually: https://ollama.com/download"
    exit 1
  fi
  export PATH="$PATH:$HOME/.local/bin:$HOME/.ollama/bin"
  hash -r 2>/dev/null || true
  command -v ollama >/dev/null 2>&1 || {
    warn "ollama not on PATH. Open a new terminal and re-run with SKIP_INSTALL=1."
    exit 1
  }
fi
ok "ollama found: $(command -v ollama)"

# --- 2. Start the server if it is not listening -----------------------------
if server_up; then
  step "Ollama server already running on :11434"
else
  step "Starting ollama serve in the background"
  nohup ollama serve >/tmp/ollama-serve.log 2>&1 &
  for _ in $(seq 1 15); do
    sleep 2
    server_up && break
  done
  server_up || { warn "Ollama server did not come up on http://localhost:11434"; exit 1; }
fi
ok "server answering at http://localhost:11434"

# --- 3. Pull the model -------------------------------------------------------
step "Ensuring model '$MODEL' is available"
if curl -sf http://localhost:11434/api/tags | grep -qF "\"name\":\"$MODEL"; then
  ok "model '$MODEL' already pulled"
else
  ollama pull "$MODEL"
fi

# --- 4. Write .env -----------------------------------------------------------
step "Writing .env (gitignored; never committed)"
cat > .env <<EOF
# Byte Analyst - local model configuration (written by scripts/setup_local_model.sh)
# No API key needed: Ollama runs on this machine.
BYTEANALYST_BASE_URL=http://localhost:11434/v1
BYTEANALYST_MODEL=$MODEL
# Loading a multi-GB model into RAM on a CPU can exceed the 120 s default.
BYTEANALYST_TIMEOUT=600
# 7-8B local models call tools far more reliably with a smaller toolbox.
# Delete this line to get the full 10-tool set.
BYTEANALYST_TOOLS=profile_dataset,run_sql,run_python,statistical_test,detect_anomalies
EOF
ok ".env written with BYTEANALYST_MODEL=$MODEL"

# --- 5. Verify end-to-end ----------------------------------------------------
if [ "${SKIP_VERIFY:-0}" = "1" ]; then
  step "Skipping verification (SKIP_VERIFY=1)"
else
  step "Verifying: byteanalyst -> Ollama -> answer"
  PYTHON=".venv/bin/python"
  [ -x "$PYTHON" ] || PYTHON=python3
  "$PYTHON" -m byteanalyst ask "How many rows are in this table? Answer in one short sentence." \
    --data data/employees.csv --no-color || {
      warn "Verification turn failed. Server and model are ready; try:"
      warn "  byteanalyst ask \"profile the data\" --data data/employees.csv"
      exit 1
    }
fi

printf '\nReady. Every session now works with:\n'
printf '  byteanalyst ask "profile the data" --data data/weather_daily.csv\n'
printf '(.env supplies the base URL and model automatically)\n'
