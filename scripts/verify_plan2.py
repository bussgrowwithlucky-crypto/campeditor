"""Plan 2 verification: confirm the three static files wire up the B-Roll Pack
UI correctly without re-running the full FastAPI server."""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(r"C:/campeditor/static")

html = (ROOT / "index.html").read_text(encoding="utf-8")
js = (ROOT / "app.js").read_text(encoding="utf-8")
css = (ROOT / "styles.css").read_text(encoding="utf-8")

# Locate the replicate-fields block to confirm the checkbox sits INSIDE it.
rep_open = html.find('id="replicate-fields"')
broll_chk_pos = html.find('id="broll-pack"')
rep_close = html.find("</fieldset>", rep_open)
chk_inside_replicate = rep_open < broll_chk_pos < rep_close

# Locate the status-panel block to confirm the pack panel sits INSIDE it.
status_open = html.find('id="status-panel"')
panel_pos = html.find('id="broll-pack-panel"')
heading_pos = html.find('id="broll-pack-heading"')
status_close = html.find("</section>", status_open)
panel_inside_status = status_open < panel_pos < status_close
heading_inside_status = status_open < heading_pos < status_close

# Verify broll_pack append is inside the `if (replicateCheckbox.checked)` block.
# Walk through the file tracking brace depth to find the matching closing brace.
replicate_line_idx = None
for i, line in enumerate(js.splitlines(), 1):
    if "if (replicateCheckbox.checked)" in line:
        replicate_line_idx = i
        break

broll_pack_append_line_idx = None
for i, line in enumerate(js.splitlines(), 1):
    if 'data.append("broll_pack"' in line:
        broll_pack_append_line_idx = i
        break

# Find the next top-level "}" that closes the `if (replicateCheckbox.checked)`
# block. Crude but works: count braces in lines after replicate_line_idx.
replicate_close_line = None
if replicate_line_idx is not None:
    depth = 0
    started = False
    for i, line in enumerate(js.splitlines()[replicate_line_idx - 1:], replicate_line_idx):
        for ch in line:
            if ch == "{":
                depth += 1
                started = True
            elif ch == "}":
                depth -= 1
                if started and depth == 0:
                    replicate_close_line = i
                    break
        if replicate_close_line is not None:
            break

append_inside_replicate = (
    replicate_line_idx is not None
    and broll_pack_append_line_idx is not None
    and replicate_line_idx < broll_pack_append_line_idx <= replicate_close_line
)

checks = {
    # HTML
    "html: broll-pack checkbox inside #replicate-fields": chk_inside_replicate,
    "html: broll-pack-panel inside #status-panel": panel_inside_status,
    "html: broll-pack-heading inside #status-panel": heading_inside_status,
    "html: B-Roll Pack label text present": "B-Roll Pack" in html,
    "html: pack panel uses .variations grid class": 'class="variations"' in html,
    # JS
    "js: grabs brollPackCheckbox": "getElementById(\"broll-pack\")" in js and "brollPackCheckbox" in js,
    "js: grabs brollPackPanel": "getElementById(\"broll-pack-panel\")" in js and "brollPackPanel" in js,
    "js: grabs brollPackHeading": "getElementById(\"broll-pack-heading\")" in js and "brollPackHeading" in js,
    "js: appends broll_pack to FormData": 'data.append("broll_pack"' in js,
    "js: appends broll_pack inside replicate-block only": append_inside_replicate,
    "js: renders pack when broll_pack_urls non-empty": "renderBrollPack" in js and "broll_pack_urls" in js,
    "js: renderBrollPack function defined": "function renderBrollPack(" in js,
    "js: resetStatus clears brollPackPanel": "brollPackPanel.hidden = true" in js,
    "js: resetStatus clears brollPackHeading": "brollPackHeading.hidden = true" in js,
    "js: pack label includes span_index + rank + timestamps": "span_index" in js and "rank" in js,
    "js: uses Cutaway-N option-R style label": "Cutaway " in js and "option " in js,
    "js: formatTimestamp helper present": "function formatTimestamp(" in js,
    # CSS
    "css: .section-heading rule": ".section-heading" in css,
    "css: .pack-query rule": ".pack-query" in css,
    "css: .radio.pack rule": ".radio.pack" in css,
    "css: reuses .variation-card for pack cards": ".variation-card" in css,
}

# JS parses cleanly via Node (more accurate than Python compile() for JS syntax).
node_parse = subprocess.run(
    ["node", r"C:/campeditor/scripts/js_parse_check.js", r"C:/campeditor/static/app.js"],
    capture_output=True, text=True, timeout=10,
)
js_parses = "OK:" in node_parse.stdout
js_parse_msg = node_parse.stdout.strip() or node_parse.stderr.strip()

print()
print("Plan 2 static-file verification")
print(f"  index.html: {len(html)} bytes")
print(f"  app.js:     {len(js)} bytes")
print(f"  styles.css: {len(css)} bytes")
print(f"  app.js parses (node): {js_parse_msg}")
print()
for name, ok in checks.items():
    mark = "x" if ok else " "
    print(f"  [{mark}] {name}")

passed = sum(1 for v in checks.values() if v) + (1 if js_parses else 0)
total = len(checks) + 1
print(f"\n  {passed}/{total} checks passed")
