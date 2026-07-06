$script = @'
<#
.SYNOPSIS
    campeditor one-shot installer + launcher.
#>

[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$HostName = "127.0.0.1",
    [switch]$SkipStart,
    [string]$ProjectRoot = ""
)

$ErrorActionPreference = "Stop"
if (-not $ProjectRoot) { $ProjectRoot = (Get-Location).Path }
Set-Location $ProjectRoot
$root = $ProjectRoot

function Write-Step($msg)  { Write-Host "" ; Write-Host ("=== {0} ===" -f $msg) -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host ("  [OK] {0}" -f $msg) -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host ("  [!!] {0}" -f $msg) -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host ("  [ERR] {0}" -f $msg) -ForegroundColor Red }

function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('Path','User')
}

function Test-PythonVersion($py) {
    try {
        $v = & $py -c "import sys; print('%d.%d.%d' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro))" 2>$null
        return $v
    } catch { return $null }
}

Write-Step "Step 1/4  Checking Python 3.11+"

$python = $null
foreach ($cmd in @("python", "py", "python3")) {
    $p = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($p) {
        $ver = Test-PythonVersion $p.Source
        if ($ver) {
            Write-Ok ("Found {0} ({1}) at {2}" -f $cmd, $ver, $p.Source)
            $python = $p.Source
            break
        }
    }
}

if (-not $python) {
    $pkgMgr = Get-Command winget -ErrorAction SilentlyContinue
    if ($pkgMgr) {
        Write-Warn "Python not found - installing via winget..."
        winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements | Out-Null
        Refresh-Path
        $python = (Get-Command python -ErrorAction SilentlyContinue).Source
    }
}

if (-not $python) {
    Write-Err "Python 3.11+ is required but not installed."
    Write-Host ""
    Write-Host "  Manual install:   https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  Or via winget:    winget install -e --id Python.Python.3.12" -ForegroundColor White
    Write-Host "  Or via chocolatey: choco install python --version=3.12.0" -ForegroundColor White
    exit 1
}

$ver = Test-PythonVersion $python
$parts = $ver.Split('.')
$majorMinor = [double]($parts[0] + '.' + $parts[1])
if ($majorMinor -lt 3.11) {
    Write-Err ("Python {0} found, but 3.11+ required." -f $ver)
    Write-Host "  Install Python 3.12 from https://www.python.org/downloads/" -ForegroundColor White
    exit 1
}

Write-Step "Step 2/4  Checking FFmpeg"

$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
if (-not $ffmpeg) {
    $pkgMgr = Get-Command winget -ErrorAction SilentlyContinue
    if ($pkgMgr) {
        Write-Warn "FFmpeg not found - installing via winget..."
        winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements | Out-Null
        Refresh-Path
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    }
}

if (-not $ffmpeg) {
    $pkgMgr = Get-Command choco -ErrorAction SilentlyContinue
    if ($pkgMgr) {
        Write-Warn "FFmpeg not found - installing via chocolatey..."
        choco install ffmpeg -y | Out-Null
        Refresh-Path
        $ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
    }
}

if (-not $ffmpeg) {
    Write-Err "FFmpeg is required but not installed."
    Write-Host ""
    Write-Host "  Manual install:   https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor White
    Write-Host "  Add the bin folder to your PATH." -ForegroundColor White
    Write-Host "  Or via winget:    winget install -e --id Gyan.FFmpeg" -ForegroundColor White
    Write-Host "  Or via chocolatey: choco install ffmpeg" -ForegroundColor White
    exit 1
}
Write-Ok ("FFmpeg found at {0}" -f $ffmpeg.Source)

Write-Step "Step 3/4  Setting up Python virtualenv + dependencies"

if (-not (Test-Path ".venv")) {
    Write-Ok "Creating .venv ..."
    & $python -m venv .venv
} else {
    Write-Ok ".venv already exists, reusing"
}

$venvPy = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Err ("Virtualenv python.exe missing at {0}" -f $venvPy)
    exit 1
}

Write-Ok "Upgrading pip ..."
& $venvPy -m pip install --upgrade pip --quiet | Out-Null

Write-Ok "Installing campeditor + dependencies (this may take a few minutes) ..."
& $venvPy -m pip install -e . --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Standard install failed, retrying with binary-only wheels ..."
    & $venvPy -m pip install -e . --only-binary=:all:
    if ($LASTEXITCODE -ne 0) {
        Write-Err "pip install failed. Check your internet connection and try again."
        exit 1
    }
}
Write-Ok "All dependencies installed"

Write-Step "Step 4/4  Setting up environment file"

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Ok "Created .env from .env.example"
        Write-Host ""
        Write-Host "  +------------------------------------------------------------+" -ForegroundColor Yellow
        Write-Host "  | Edit .env and paste your own API keys before continuing.     |" -ForegroundColor Yellow
        Write-Host "  | See FRIEND_README.md for the list of services + signup URLs.|" -ForegroundColor Yellow
        Write-Host "  +------------------------------------------------------------+" -ForegroundColor Yellow
    } else {
        Write-Warn ".env.example not found - continuing without .env"
    }
} else {
    Write-Ok ".env already exists"
}

if ($SkipStart) {
    Write-Host ""
    Write-Host "Install complete. Run this to start the server later:" -ForegroundColor Cyan
    Write-Host "    .\.venv\Scripts\python.exe -m uvicorn app.main:app --host $HostName --port $Port" -ForegroundColor White
    exit 0
}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ("  campeditor is starting at http://{0}:{1}" -f $HostName, $Port) -ForegroundColor Green
Write-Host "  Open that URL in your browser. Press Ctrl+C to stop." -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""

& $venvPy -m uvicorn app.main:app --host $HostName --port $Port
'@
$tmp = "$env:TEMP\campeditor_install.ps1"
[System.IO.File]::WriteAllText($tmp, $script, (New-Object System.Text.UTF8Encoding($false)))
Write-Host ("Wrote installer to {0}" -f $tmp) -ForegroundColor Green
Write-Host "Launching installer against current dir: $(Get-Location)" -ForegroundColor Cyan
powershell -ExecutionPolicy Bypass -File $tmp -ProjectRoot (Get-Location).Path