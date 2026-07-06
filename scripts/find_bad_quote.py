"""Find unterminated-string culprits in app.js."""
import re
src = open(r"C:/campeditor/static/app.js", encoding="utf-8").read()
for i, line in enumerate(src.splitlines(), 1):
    if '"' not in line:
        continue
    # Count unescaped double quotes (skip \\ and \").
    n = 0
    j = 0
    while j < len(line):
        c = line[j]
        if c == "\\" and j + 1 < len(line):
            j += 2
            continue
        if c == '"':
            n += 1
        j += 1
    if n % 2 == 1:
        print(f"L{i} odd-quote ({n}): {line[:100]!r}")
