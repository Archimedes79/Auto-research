"""Rebuild the files that a rollback removed, third version: replay of the Write / Edit tool calls AND of the shell commands that patched the files (node -e, heredoc appends, sed -i), executed inside a sandbox tree (paths rewritten to the staging tree, wsl / nvidia-smi neutralised).
Nothing is copied into place. Usage: _recover3.py <transcript.jsonl> <staging dir>"""
import json
import os
import re
import stat
import subprocess
import sys

src, stage = sys.argv[1], os.path.abspath(sys.argv[2])
R = os.path.join(stage, "tree")
FB = os.path.join(stage, "fakebin")
os.makedirs(R, exist_ok=True)
os.makedirs(FB, exist_ok=True)
for nm, body in (("wsl.exe", "exit 0"), ("wsl", "exit 0"), ("nvidia-smi", "exit 0"),
                 ("node", 'a=()\nfor x in "$@"; do a+=("${x//\\/mnt\\/c\\//C:/}"); done\nexec "/mnt/c/Program Files/nodejs/node.exe" "${a[@]}"')):
    p = os.path.join(FB, nm)
    open(p, "w", newline="\n").write("#!/bin/bash\n" + body + "\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)
P = "C:\\ML-Experiment\\"
calls, results, SEEN = [], {}, set()
with open(src, encoding="utf-8", errors="replace") as f:
    for ln, line in enumerate(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("type") not in ("assistant", "user"):  # skip the api-request-blob copies of earlier messages (they would replay a tool call twice)
            continue
        m = d.get("message")
        if not isinstance(m, dict) or not isinstance(m.get("content"), list):
            continue
        for b in m["content"]:
            if not isinstance(b, dict):
                continue
            if m.get("role") == "assistant" and b.get("type") == "tool_use":
                if b.get("id") not in SEEN:
                    SEEN.add(b.get("id"))
                    calls.append((ln, b))
            elif m.get("role") == "user" and b.get("type") == "tool_result":
                results[b.get("tool_use_id")] = b
files, log = {}, []
DEBUG = []
tracked_names = set()


def disk(fp):
    return os.path.join(R, fp.replace(P, "").replace("\\", "/"))


def sync_out():
    for fp, c in files.items():
        o = disk(fp)
        os.makedirs(os.path.dirname(o), exist_ok=True)
        open(o, "w", encoding="utf-8", newline="").write(c)


def sync_in(ln):
    for fp in list(files):
        o = disk(fp)
        if os.path.exists(o):
            c = open(o, encoding="utf-8", newline="").read()
            if c != files[fp]:
                files[fp] = c
                log.append((ln, fp, "changed by shell command"))


DANGER = re.compile(r"\bkill\b|pkill|shutdown|Remove-Item|rm -rf|nohup|python|mkfs|\bdd\b")
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
                tracked_names.add(os.path.basename(fp.replace("\\", "/")))
            log.append((ln, fp, "Write ok" if ok else "Write FAILED"))
        else:
            if not ok:
                log.append((ln, fp, "Edit failed in the original (skipped)"))
                continue
            old = inp.get("old_string")
            new = inp.get("new_string", inp.get("new_str"))
            if old is None or new is None:
                log.append((ln, fp, "Edit UNKNOWN SHAPE " + ",".join(inp.keys())))
                continue
            if fp not in files:
                log.append((ln, fp, "Edit NO BASE"))
                continue
            if old not in files[fp]:
                import difflib

                first = [x for x in old.splitlines() if x.strip()][:1]
                cand = difflib.get_close_matches(first[0], files[fp].splitlines(), n=1, cutoff=0.5) if first else []
                log.append((ln, fp, "Edit OLD NOT FOUND (out of sync)"))
                DEBUG.append((ln, os.path.basename(fp.replace("\\", "/")), first[0][:110] if first else "", cand[0][:110] if cand else "<<no close line in the replayed file>>"))
                continue
            files[fp] = files[fp].replace(old, new) if inp.get("replace_all") else files[fp].replace(old, new, 1)
            log.append((ln, fp, "Edit ok"))
    elif nm == "Bash":
        cmd = inp.get("command", "")
        if not any(n in cmd for n in tracked_names):
            continue
        if not ("node -e" in cmd or re.search(r"cat >> \S+ <<'PYEOF'", cmd) or "sed -i" in cmd):
            continue
        if DANGER.search(cmd.replace("py_compile", "")):
            log.append((ln, "-", "SKIPPED shell command (danger pattern): " + re.sub(r"\s+", " ", cmd)[:90]))
            continue
        sync_out()
        c2 = cmd.replace("/mnt/c/ML-Experiment", "@@R@@").replace("/c/ML-Experiment", "@@R@@").replace("@@R@@", R.replace("\\", "/"))
        env = dict(os.environ, PATH=FB + ":" + os.environ.get("PATH", ""))
        pr = subprocess.run(["bash", "-c", c2], cwd=R, env=env, capture_output=True, text=True, timeout=120)
        miss = [x for x in (pr.stdout + pr.stderr).splitlines() if "MISSING" in x]
        log.append((ln, "-", "shell command run" + (" MISSING-reports: " + "; ".join(miss) if miss else "")))
        sync_in(ln)
bad = [x for x in log if not any(k in str(x[2]) for k in ("Write ok", "Edit ok", "shell command run", "changed by shell", "skipped"))]
bad += [x for x in log if "MISSING-reports" in str(x[2]) or "SKIPPED" in str(x[2])]
print("steps:", len(log), "problems:", len(bad))
for x in bad:
    print("  PROBLEM", x)
sync_out()
for x in DEBUG:
    print("  DEBUG", x)
print("files rebuilt:", len(files), "in", R)
json.dump(log, open(os.path.join(stage, "_replay_log3.json"), "w"), indent=0)
