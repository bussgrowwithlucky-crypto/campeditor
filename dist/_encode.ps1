$src = 'C:\campeditor\dist\installer_inline.ps1'
$content = Get-Content -Path $src -Raw
$bytes = [System.Text.Encoding]::Unicode.GetBytes($content)  # UTF-16LE, no BOM
$encoded = [Convert]::ToBase64String($bytes)
$line = "powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand $encoded"
$line | Set-Content -Path 'C:\campeditor\dist\one_liner.txt' -Encoding UTF8
Write-Host ("One-liner length: {0} chars" -f $line.Length)
Write-Host ("Base64 length:    {0} chars" -f $encoded.Length)
Write-Host ""
Write-Host "First 200 chars of one-liner:"
Write-Host $line.Substring(0, [Math]::Min(200, $line.Length))
Write-Host "..."
Write-Host "Last 100 chars:"
Write-Host $line.Substring([Math]::Max(0, $line.Length - 100))