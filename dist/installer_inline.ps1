[CmdletBinding()]
param([int]$Port=8000,[string]$HostName="127.0.0.1",[switch]$SkipStart,[string]$ProjectRoot="")
$ErrorActionPreference="Stop"
if(-not $ProjectRoot){$ProjectRoot=(Get-Location).Path}
Set-Location $ProjectRoot
$root=$ProjectRoot
function Step($m){Write-Host "";Write-Host ("=== {0} ===" -f $m) -ForegroundColor Cyan}
function Ok($m){Write-Host ("  [OK] {0}" -f $m) -ForegroundColor Green}
function Warn($m){Write-Host ("  [!!] {0}" -f $m) -ForegroundColor Yellow}
function Err($m){Write-Host ("  [ERR] {0}" -f $m) -ForegroundColor Red}
function RefreshPath{$env:Path=[Environment]::GetEnvironmentVariable('Path','Machine')+';'+[Environment]::GetEnvironmentVariable('Path','User')}
function PyVer($py){try{& $py -c "import sys;print('%d.%d.%d'%(sys.version_info.major,sys.version_info.minor,sys.version_info.micro))" 2>$null}catch{}}

Step "Step 1/4  Checking Python 3.11+"
$python=$null
foreach($c in @("python","py","python3")){$p=Get-Command $c -EA SilentlyContinue;if($p){$v=PyVer $p.Source;if($v){Ok ("Found {0} ({1}) at {2}" -f $c,$v,$p.Source);$python=$p.Source;break}}}
if(-not $python){$w=Get-Command winget -EA SilentlyContinue;if($w){Warn "Python not found - installing via winget...";winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements | Out-Null;RefreshPath;$python=(Get-Command python -EA SilentlyContinue).Source}}
if(-not $python){Err "Python 3.11+ required but missing.";Write-Host "Manual: https://www.python.org/downloads/" -ForegroundColor White;exit 1}
$ver=PyVer $python;$pp=$ver.Split('.');$mm=[double]($pp[0]+'.'+$pp[1])
if($mm -lt 3.11){Err "Python $ver found, but 3.11+ required.";exit 1}

Step "Step 2/4  Checking FFmpeg"
$ff=Get-Command ffmpeg -EA SilentlyContinue
if(-not $ff){$w=Get-Command winget -EA SilentlyContinue;if($w){Warn "FFmpeg missing - installing via winget...";winget install -e --id Gyan.FFmpeg --accept-package-agreements --accept-source-agreements | Out-Null;RefreshPath;$ff=Get-Command ffmpeg -EA SilentlyContinue}}
if(-not $ff){$c=Get-Command choco -EA SilentlyContinue;if($c){Warn "FFmpeg missing - installing via chocolatey...";choco install ffmpeg -y | Out-Null;RefreshPath;$ff=Get-Command ffmpeg -EA SilentlyContinue}}
if(-not $ff){Err "FFmpeg required but missing.";Write-Host "Manual: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor White;exit 1}
Ok ("FFmpeg at {0}" -f $ff.Source)

Step "Step 3/4  Setting up virtualenv + dependencies"
if(-not (Test-Path ".venv")){Ok "Creating .venv ...";& $python -m venv .venv}else{Ok ".venv exists, reusing"}
$venvPy=Join-Path $root ".venv\Scripts\python.exe"
if(-not (Test-Path $venvPy)){Err "venv python missing";exit 1}
Ok "Upgrading pip ...";& $venvPy -m pip install --upgrade pip --quiet | Out-Null
Ok "Installing campeditor + dependencies (this takes a few minutes)..."
& $venvPy -m pip install -e . --quiet
if($LASTEXITCODE -ne 0){Warn "Retrying with binary wheels...";& $venvPy -m pip install -e . --only-binary=:all:;if($LASTEXITCODE -ne 0){Err "pip install failed";exit 1}}
Ok "All dependencies installed"

Step "Step 4/4  Setting up .env"
if(-not (Test-Path ".env")){if(Test-Path ".env.example"){Copy-Item ".env.example" ".env";Ok "Created .env from .env.example";Write-Host "";Write-Host "  Edit .env with your API keys before first use." -ForegroundColor Yellow}else{Warn ".env.example missing"}}
else{Ok ".env already exists"}

if($SkipStart){Write-Host "";Write-Host "Install complete. To run:" -ForegroundColor Cyan;Write-Host "  .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000" -ForegroundColor White;exit 0}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ("  campeditor starting at http://{0}:{1}" -f $HostName,$Port) -ForegroundColor Green
Write-Host "  Open that URL. Ctrl+C to stop." -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
& $venvPy -m uvicorn app.main:app --host $HostName --port $Port