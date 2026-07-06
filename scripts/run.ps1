param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1"
)

$ErrorActionPreference = "Stop"

Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .[test]
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host $HostName --port $Port
