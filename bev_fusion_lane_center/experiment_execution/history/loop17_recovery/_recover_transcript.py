"""Rebuild files that a rollback removed (files created or edited with the Write / Edit tools of a session): replay the Write and Edit tool calls of the session transcript in order and write the results to a STAGING folder
(nothing is copied into place here). Bash / PowerShell commands that mention a rebuilt file with a modifying pattern are listed for review.
Usage: _recover_transcript.py <transcript.jsonl> <staging dir>"""
import json
import os
import re
import sys

src, stage = sys.argv[1], sys.argv[2]
os.makedirs(stage, exist_ok=True)
calls, results = [], {}
with open(src, encoding="utf-8", errors="replace") as f:
    for ln, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("message")
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            if m.get("role") == "assistant" and b.get("type") == "tool_use":
                calls.append((ln, b))
            elif m.get("role") == "user" and b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = b
files = {}
log = []
bash_mods = []
tracked = set()
for ln, b in calls:
    nm, inp = b.get("name"), b.get("input", {})
    fp = inp.get("file_path")
    if nm in ("Write", "Edit") and fp and "ML-Experiment" in fp:
        r = results.get(b["id"])
        err = bool(r and r.get("is_error"))
        text = ""
        if r:
            cc = r.get("content")
            text = cc if isinstance(cc, str) else " ".join(x.get("text", "") for x in cc if isinstance(x, dict)) if isinstance(cc, list) else ""
        ok = (not err) and not text.startswith("<tool_use_error>")
        if nm == "Write":
            if ok:
                files[fp] = inp["content"]
                tracked.add(fp)
                log.append((ln, "Write", fp, "ok"))
            else:
                log.append((ln, "Write", fp, "FAILED " + text[:80]))
        else:
            if not ok:
                log.append((ln, "Edit", fp, "FAILED (skipped) " + text[:80]))
                continue
            if fp not in files:
                log.append((ln, "Edit", fp, "NO BASE CONTENT (file not written by Write in this transcript)"))
                continue
            if "old_string" not in inp or "new_string" not in inp:
                log.append((ln, "Edit", fp, "UNKNOWN SHAPE keys=" + ",".join(inp.keys())))
                continue
            old, new, al = inp["old_string"], inp["new_string"], inp.get("replace_all", False)
            if old not in files[fp]:
                log.append((ln, "Edit", fp, "OLD STRING NOT FOUND in the replay (out of sync)"))
                continue
            files[fp] = files[fp].replace(old, new) if al else files[fp].replace(old, new, 1)
            log.append((ln, "Edit", fp, "ok"))
    elif nm in ("Bash", "PowerShell"):
        cmd = inp.get("command", "")
        for t in list(tracked):
            base = os.path.basename(t.replace("\\", "/"))
            if base in cmd and re.search(r"sed -i|>>|\bcat\s*>|tee |Set-Content|Add-Content|Out-File|open\([^)]*['\"]w|\.write\(|mv |cp |rm |Remove-Item|truncate|os\.replace", cmd):
                bash_mods.append((ln, base, cmd[:160].replace("\n", " ")))
bad = [x for x in log if x[3] != "ok"]
print("tool calls replayed:", len(log), "problems:", len(bad))
for x in bad:
    print("  PROBLEM", x)
for fp, content in files.items():
    rel = fp.replace("C:\\ML-Experiment\\", "").replace("\\", "__")
    open(os.path.join(stage, rel), "w", encoding="utf-8", newline="").write(content)
print("files rebuilt in", stage, ":", len(files))
print("Bash / PowerShell commands that mention a rebuilt file with a modifying pattern:", len(bash_mods))
for x in bash_mods[:60]:
    print("  ", x)
json.dump([list(map(str, x)) for x in log], open(os.path.join(stage, "_replay_log.json"), "w"))
