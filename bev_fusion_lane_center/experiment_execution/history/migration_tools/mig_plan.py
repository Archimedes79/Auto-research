"""Dry run of the migration: decides for every script and every out/loop2 entry where it goes. Writes restructure/plan.json, touches nothing else.
Run in WSL: _py14.sh ../restructure/mig_plan.py"""
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403

inv = json.load(open(f"{RS}/inventory.json"))
S = {s["name"]: s for s in inv["scripts"]}
PY = {n[:-3]: s for n, s in S.items() if n.endswith(".py")}
RECENT = 1761177600  # 2026-10-23 00:00 UTC-ish; loops 15-17 live code is newer than the 22nd, see below
import calendar
import time

RECENT = calendar.timegm(time.strptime("2026-10-22", "%Y-%m-%d"))

MINE = {"_inventory.py", "_classify.py", "_recover3.py", "_recover2.py", "_recover_transcript.py", "_survey_transcript.py", "_dump_cmds.py", "_scan_cmds.py"}
ROOT_PAT = re.compile(r"^(p1[3-7]_|e2e_|loop1[4-7]|build_loop1[4-7]|fig_p1[4-7]|graph_eval|report_lib|arch_figs)")


def is_junk(n):
    return bool(JUNK_RE.match(n)) or bool(BAK_RE.search(n)) or n == "__pycache__"


# ---------------------------------------------------------------- live set (python)
live = set()
todo = [m for m in PY if ROOT_PAT.match(m)] + [m for m, s in PY.items() if s["imported_by"] >= 2]
todo += [m for m, s in PY.items() if s["mtime"] >= RECENT and not is_junk(m + ".py") and m + ".py" not in MINE]
while todo:
    m = todo.pop()
    if m in live or m not in PY:
        continue
    live.add(m)
    todo.extend(PY[m]["imports"])

# ---------------------------------------------------------------- table rows -> experiment folders
rows = load_table()
row_of_id = {}
folders = []
for r in rows:
    fn = folder_name(r)
    r["folder"] = fn
    folders.append(fn)
    for n in ids_of_row(r["id"]):
        row_of_id.setdefault(n, fn)

# ids that occur in out dirs / script names but in no table row -> extra folders
extra = {
    123: "E123_differentiable-ladder-of-eight-builders",
}
for k, v in extra.items():
    row_of_id.setdefault(k, v)
    if v not in folders:
        folders.append(v)

# ---------------------------------------------------------------- scripts
EVAL_SPARSE = {"graph_eval.py"}
TOOLS_VIZ = {"viz_graph.py"}
dest = {}
why = {}
for n, s in S.items():
    b = n[:-3] if n.endswith(".py") else None
    if n in MINE or n.startswith(("_inventory", "_classify")):
        dest[n], why[n] = "tools/restructure", "migration tooling"
    elif n in EVAL_SPARSE:
        dest[n], why[n] = "evaluation/sparse_lanes", "sparse graph metric"
    elif n in TOOLS_VIZ:
        dest[n], why[n] = "tools/viz", "graph drawing"
    elif n == "__pycache__":
        dest[n], why[n] = f"_to_delete/scripts_junk", "bytecode cache"
    elif b and b in live:
        dest[n], why[n] = "scripts", "live code (closure of loop 13-17 roots, shared libs, recent files)"
    elif JOBCTL_RE.match(n) or BAK_RE.search(n):
        dest[n], why[n] = "_to_delete/scripts_junk", "job control / backup copy / lock"
    elif JUNK_RE.match(n):
        dest[n], why[n] = "_to_delete/scripts_junk", "one-off patch / debug helper"
    elif n.endswith((".sh", ".ps1")) and S[n]["mtime"] >= RECENT:
        dest[n], why[n] = "scripts", "recent run wrapper"
    else:
        ids = eids_in_script(n)
        fol = next((row_of_id[i] for i in ids if i in row_of_id), None)
        if fol:
            dest[n], why[n] = f"experiments/{fol}/code", f"names experiment E{ids[0]}"
        else:
            g = archive_group(n)
            if g == "tools_env":
                dest[n], why[n] = "tools/env", "environment / data setup"
            else:
                dest[n], why[n] = f"scripts/_archive/{g}", "historic, not part of the live closure"

# ---------------------------------------------------------------- tokens that keep run dirs alive
tok = set()


def read(p):
    try:
        return open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


for n, d in dest.items():
    if d in ("scripts", "evaluation/sparse_lanes", "tools/viz") and n.endswith((".py", ".sh", ".ps1")):
        tok.update(re.findall(r"\be\d{2,3}[a-z0-9_]*", read(f"{O}/scripts/{n}")))
for doc in (f"{O}/reports/Loop_Prompt_Next_EN.md", f"{O}/loop2/STATE.md", f"{O}/out/loop2/e123_ladder/TRIAGE_STATE.md", f"{O}/out/loop2/LOOP15_INDEX.md"):
    tok.update(re.findall(r"\be\d{2,3}[a-z0-9_]*", read(doc)))
tok = {t for t in tok if len(t) >= 3}

# ---------------------------------------------------------------- out/loop2 entries
def matched(nm):
    base = nm.rsplit(".", 1)[0] if nm.endswith((".done", ".log", ".out")) else nm
    for t in tok:
        if base == t or nm.startswith(t + "."):
            return True
        if t.endswith("_") and nm.startswith(t):
            return True
    return False


def matched_by(nm):
    base = nm.rsplit(".", 1)[0] if nm.endswith((".done", ".log", ".out")) else nm
    return [t for t in tok if base == t or nm.startswith(t + ".") or (t.endswith("_") and nm.startswith(t))][:3]


runs = {}
for r in inv["out"]:
    nm = r["name"]
    eid = r["eid"]
    keep, rsn = False, ""
    if nm in ("EXPERIMENT_TABLE.csv", "EXPERIMENT_TABLE.json", "FINAL_NAMES.json", "splits.json", "_scene_ids.pkl") or nm.startswith(("LOOP", "e123_ladder")):
        keep, rsn = True, "state / index / ladder"
    elif eid is not None and eid >= 117:
        keep, rsn = True, "loop 14-17 (live chain)"
    elif eid in (94, 120, 123):
        keep, rsn = True, "control / champion chain"
    elif matched(nm):
        keep, rsn = True, "named by live code or state files"
    elif r["kind"] in ("log", "build") or nm.endswith((".out", ".done", ".log")):
        keep, rsn = False, "log / build artefact"
    elif r["kind"] == "debug":
        keep, rsn = False, "debug / smoke run"
    else:
        keep, rsn = False, "historic run, not referenced by live code"
    fol = row_of_id.get(eid) if eid is not None else None
    runs[nm] = dict(keep=keep, why=rsn, mb=r["mb"], heavy_mb=r["heavy_mb"], eid=eid, exp=fol, isdir=r["isdir"])

plan = dict(scripts=dest, why=why, runs=runs, folders=folders, row_of_id={str(k): v for k, v in row_of_id.items()}, live=sorted(live))
json.dump(plan, open(f"{RS}/plan.json", "w"), indent=0)

# ---------------------------------------------------------------- summary
c = Counter(v.split("/")[0] + ("/" + v.split("/")[1] if v.startswith(("scripts/_archive", "_to_delete")) else "") for v in dest.values())
print("scripts by destination:", dict(c))
exp_n = sum(1 for v in dest.values() if v.startswith("experiments/"))
print("scripts into experiment code folders:", exp_n, "in", len({v for v in dest.values() if v.startswith('experiments/')}), "folders")
keepmb = sum(r["mb"] for r in runs.values() if r["keep"])
stagemb = sum(r["mb"] for r in runs.values() if not r["keep"])
print(f"runs kept: {sum(1 for r in runs.values() if r['keep'])} entries {keepmb/1000:.1f} GB; staged: {sum(1 for r in runs.values() if not r['keep'])} entries {stagemb/1000:.1f} GB")
by = defaultdict(lambda: [0, 0.0])
for r in runs.values():
    by[r["why"]][0] += 1
    by[r["why"]][1] += r["mb"]
for k, v in sorted(by.items(), key=lambda kv: -kv[1][1]):
    print(f"   {k:48s} {v[0]:5d} entries {v[1]/1000:8.1f} GB")
noexp = [nm for nm, r in runs.items() if r["eid"] is not None and not r["exp"]]
print("run entries with an E id but no table row:", len(noexp), sorted({runs[n]['eid'] for n in noexp}))
print("experiment folders:", len(folders))
miss = Counter()
for n, d in dest.items():
    if d.startswith("scripts/_archive"):
        miss[d.split("/")[-1]] += 1
print("archive groups:", dict(miss))
big_staged = sorted(((r["mb"], nm) for nm, r in runs.items() if not r["keep"]), reverse=True)[:12]
print("largest staged:", [(nm, round(mb)) for mb, nm in big_staged])
big_kept0 = sorted(((r["mb"], nm) for nm, r in runs.items() if r["keep"]), reverse=True)[:25]

big_kept = sorted(((r["mb"], nm) for nm, r in runs.items() if r["keep"] and r["why"].startswith("named")), reverse=True)[:30]
print("kept by token (largest):", [(nm, round(mb), matched_by(nm)) for mb, nm in big_kept])
