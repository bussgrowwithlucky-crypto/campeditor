$line = (Get-Content 'C:\campeditor\dist\one_liner.txt' -Raw).Trim()
$base64 = $line -replace '^powershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand ', ''
$bat = "@echo off`r`nREM campeditor one-click installer. Double-click or run from any terminal.`r`ncd /d %~dp0`r`npowershell -ExecutionPolicy Bypass -NoProfile -EncodedCommand $base64"
[System.IO.File]::WriteAllText('C:\campeditor\dist\install.bat', $bat, (New-Object System.Text.UTF8Encoding($true)))
"Bytes: $((Get-Item 'C:\campeditor\dist\install.bat').Length)"
"Lines: $((Get-Content 'C:\campeditor\dist\install.bat' | Measure-Object Lines).Lines)"
"First line: $((Get-Content 'C:\campeditor\dist\install.bat' -TotalCount 1))"