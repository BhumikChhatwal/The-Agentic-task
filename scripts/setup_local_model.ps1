<#
.SYNOPSIS
    One-command setup for running Byte Analyst with a local Ollama model.

.DESCRIPTION
    Installs Ollama (if missing), pulls the model, writes .env pointing at the
    local server, and verifies the whole chain: server -> model -> byteanalyst.

.EXAMPLE
    .\scripts\setup_local_model.ps1                 # qwen2.5:7b, default install
    .\scripts\setup_local_model.ps1 -Model llama3.1    # different model
    .\scripts\setup_local_model.ps1 -SkipInstall        # Ollama already installed
#>
param(
    [string]$Model = "qwen2.5:7b",
    [switch]$SkipInstall,
    [switch]$SkipVerify
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Step($msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "    OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "    !!  $msg" -ForegroundColor Yellow }

# --- 1. Install Ollama -------------------------------------------------------
if ($SkipInstall) {
    Step "Skipping install (-SkipInstall)"
} elseif (Get-Command ollama -ErrorAction SilentlyContinue) {
    Step "Ollama already installed"
} else {
    Step "Installing Ollama via winget (several minutes, ~1 GB download)"
    winget install Ollama.Ollama --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget install failed with exit code $LASTEXITCODE" }
    # winget installs per-user; refresh this session's PATH from the registry
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path    = "$machinePath;$userPath"
    if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
        Warn "ollama not on PATH yet. Close and reopen PowerShell, then re-run this script with -SkipInstall."
        exit 1
    }
}
Ok "ollama found: $(Get-Command ollama | Select-Object -ExpandProperty Source)"

# --- 2. Start the server if it is not listening -----------------------------
function Test-OllamaServer {
    try {
        Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 | Out-Null
        return $true
    } catch { return $false }
}

if (Test-OllamaServer) {
    Step "Ollama server already running on :11434"
} else {
    Step "Starting ollama serve in the background"
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    $up = $false
    foreach ($i in 1..15) {
        Start-Sleep -Seconds 2
        if (Test-OllamaServer) { $up = $true; break }
    }
    if (-not $up) { throw "Ollama server did not come up on http://localhost:11434" }
}
Ok "server answering at http://localhost:11434"

# --- 3. Pull the model -------------------------------------------------------
$models = (Invoke-RestMethod -Uri "http://localhost:11434/api/tags").models.name
if ($models -contains $Model) {
    Step "Model '$Model' already pulled"
} else {
    Step "Pulling '$Model' (a few GB the first time)"
    ollama pull $Model
    if ($LASTEXITCODE -ne 0) { throw "ollama pull failed with exit code $LASTEXITCODE" }
}
Ok "model ready: $Model"

# --- 4. Write .env -----------------------------------------------------------
Step "Writing .env (gitignored; never committed)"
@"
# Byte Analyst - local model configuration (written by scripts/setup_local_model.ps1)
# No API key needed: Ollama runs on this machine.
BYTEANALYST_BASE_URL=http://localhost:11434/v1
BYTEANALYST_MODEL=$Model
# Loading a multi-GB model into RAM on a CPU can exceed the 120 s default.
BYTEANALYST_TIMEOUT=600
# 7-8B local models call tools far more reliably with a smaller toolbox.
# Delete this line to get the full 10-tool set.
BYTEANALYST_TOOLS=profile_dataset,run_sql,run_python,statistical_test,detect_anomalies
"@ | Set-Content -Path .env -Encoding ascii
Ok ".env written with BYTEANALYST_MODEL=$Model"

# --- 5. Verify end-to-end ----------------------------------------------------
if ($SkipVerify) {
    Step "Skipping verification (-SkipVerify)"
} else {
    Step "Verifying: byteanalyst -> Ollama -> answer"
    $env:BYTEANALYST_BASE_URL = "http://localhost:11434/v1"
    $env:BYTEANALYST_MODEL    = $Model
    & .\.venv\Scripts\python.exe -m byteanalyst ask "How many rows are in this table? Answer in one short sentence." `
        --data data/employees.csv --no-color
    if ($LASTEXITCODE -ne 0) {
        Warn "Verification turn failed (exit $LASTEXITCODE). The server and model are ready; re-run:"
        Warn "  byteanalyst ask `"profile the data`" --data data/employees.csv"
        exit 1
    }
}

Write-Host ""
Write-Host "Ready. Every session now works with:" -ForegroundColor Green
Write-Host '  byteanalyst ask "profile the data" --data data/weather_daily.csv'
Write-Host "(.env supplies the base URL and model automatically)"
