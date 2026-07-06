<#
.SYNOPSIS
    Package campeditor into a clean zip ready to hand off to your friend.
.DESCRIPTION
    Creates .\dist\campeditor.zip with everything needed EXCEPT:
      - .venv           (heavy, friend installs fresh)
      - .env            (real API keys, never share)
      - data/           (17 GB of cached downloads + debug output)
      - logs/           (server logs)
      - __pycache__/    (.pyc)
      - *.egg-info/
      - .pytest_cache/
      - .mavis/         (Mavis session data - internal)
      - .claude/        (Claude session data - internal)
      - dist/           (this script's own output)
      - broll_intelligence/ (large cached analysis data)

    Usage:
        powershell -ExecutionPolicy Bypass -File .\package_for_friend.ps1
#>

[CmdletBinding()]
param(
    [string]$OutName = "campeditor.zip"
)

$ErrorActionPreference = "Stop"
$root  = Split-Path -Parent $PSCommandPath
$dist  = Join-Path $root "dist"
$stage = Join-Path $dist "campeditor_stage"
$zip   = Join-Path $dist  $OutName

# Patterns we never ship
$excludeDirs = @(
    ".venv", ".env", "data", "logs", "__pycache__",
    "*.egg-info", ".pytest_cache", ".mavis", ".claude",
    "dist", "broll_intelligence"
)

# prep
if (Test-Path $stage) { Remove-Item $stage -Recurse -Force }
if (Test-Path $zip)   { Remove-Item $zip -Force }
if (-not (Test-Path $dist)) { New-Item -ItemType Directory -Path $dist -Force | Out-Null }
New-Item -ItemType Directory -Path $stage -Force | Out-Null

# Safety check: refuse if .env contains real-looking keys
$envFile = Join-Path $root ".env"
if (Test-Path $envFile) {
    $content = Get-Content $envFile -Raw -ErrorAction SilentlyContinue
    $suspicious = $false
    foreach ($pattern in @('gsk_[A-Za-z0-9]+', 'nvapi-[A-Za-z0-9_-]+', 'sk-nry-[A-Za-z0-9_-]+', 'AIzaSy[A-Za-z0-9_-]+')) {
        if ($content | Select-String -Pattern $pattern -Quiet) { $suspicious = $true; break }
    }
    if ($suspicious) {
        Write-Host "WARNING: Real-looking API keys detected in your .env!" -ForegroundColor Red
        Write-Host "    .env is excluded from the zip, but rotate those keys before sharing" -ForegroundColor Red
        Write-Host "    (they appeared in plaintext previously per the project README)." -ForegroundColor Red
        Write-Host ""
    }
}

# Stage files using robocopy for speed + built-in exclusions
Write-Host "Staging files (this takes a moment)..." -ForegroundColor Cyan

$robomore   = @("/MIR", "/R:1", "/W:1", "/NFL", "/NDL", "/NP", "/MT:8")
$exclusions = @("/XF", ".env") + @("/XD") + $excludeDirs

& robocopy $root $stage @($robomore + $exclusions) | Out-Null
# Robocopy exit codes 0..7 are success-ish; 8+ are failures
if ($LASTEXITCODE -ge 8) {
    throw ("Robocopy failed with exit code {0}" -f $LASTEXITCODE)
}

# Zip
Write-Host "Zipping ..." -ForegroundColor Cyan
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $stage, $zip,
    [System.IO.Compression.CompressionLevel]::Optimal,
    $false
)

# Cleanup
Remove-Item $stage -Recurse -Force

$size = [math]::Round((Get-Item $zip).Length / 1MB, 2)
Write-Host ""
Write-Host ("  [OK] Created {0}" -f $zip) -ForegroundColor Green
Write-Host ("    Size: {0} MB" -f $size) -ForegroundColor Gray
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host '  1. Send campeditor.zip to your friend (USB, Google Drive, OneDrive, WeTransfer, etc.)'
Write-Host '  2. Friend unzips it, opens the campeditor folder'
Write-Host '  3. Friend right-clicks install_and_run.ps1 -> Run with PowerShell'
Write-Host '     (or from PowerShell: powershell -ExecutionPolicy Bypass -File .\install_and_run.ps1)'