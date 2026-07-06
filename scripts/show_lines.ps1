$path = "C:/campeditor/app/jobs.py"
$lines = Get-Content $path
for ($i = 299; $i -lt 312 -and $i -lt $lines.Count; $i++) {
  Write-Host ("L{0}: {1}" -f ($i+1), $lines[$i])
}