# Start a fresh uvicorn on port 8010, capture logs.
$logDir = "C:\campeditor\logs"
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

$py = "C:\campeditor\.venv\Scripts\python.exe"
$proc = Start-Process -FilePath $py -ArgumentList @(
    "-m", "uvicorn", "app.main:app",
    "--host", "127.0.0.1", "--port", "8010",
    "--log-level", "info"
) -WorkingDirectory "C:\campeditor" `
  -RedirectStandardOutput (Join-Path $logDir "uvicorn-8010.out.log") `
  -RedirectStandardError (Join-Path $logDir "uvicorn-8010.err.log") `
  -PassThru -WindowStyle Hidden

Write-Host "Started uvicorn PID=$($proc.Id)"
Start-Sleep -Seconds 5

Write-Host ""
Write-Host "=== Process state ==="
Get-Process -Id $proc.Id -ErrorAction SilentlyContinue |
    Select-Object Id, StartTime,
        @{n="Threads";e={$_.Threads.Count}},
        @{n="Handles";e={$_.HandleCount}} |
    Format-Table

Write-Host "=== Listening on 8010 ==="
Get-NetTCPConnection -State Listen -LocalAddress 127.0.0.1 -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -eq 8010 } |
    Select-Object LocalPort, OwningProcess, State |
    Format-Table

Write-Host "=== Startup log ==="
if (Test-Path (Join-Path $logDir "uvicorn-8010.out.log")) {
    Get-Content (Join-Path $logDir "uvicorn-8010.out.log") -Tail 10
}
if (Test-Path (Join-Path $logDir "uvicorn-8010.err.log")) {
    Write-Host "--- stderr ---"
    Get-Content (Join-Path $logDir "uvicorn-8010.err.log") -Tail 15
}