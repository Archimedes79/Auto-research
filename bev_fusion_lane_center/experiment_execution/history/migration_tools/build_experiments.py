"""Build experiments/<id>_<slug>/ for every experiment of the catalogue: README.md (what, result, links), results/ (small json / csv / txt / log files, copied), figures/ (png, copied), code/ (already moved).
Raw outputs stay where they are (runs/ for kept ones, _to_delete/runs/ for staged ones); only files <= 1.5 MB of the allowed types are copied, at most 8 MB per experiment.
Run: _run.sh build_experiments.py [--go]"""
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mig_common import *  # noqa: E402,F401,F403

GO = "--go" in sys.argv
EXPD = f"{R}/experiments"
RUNS = f"{R}/runs"
STG = f"{R}/_to_delete/runs"
OKEXT = {".json", ".csv", ".txt", ".md", ".log", ".png", ".jpg"}
MAXF, MAXTOT = 1_500_000, 8_000_000

tab = json.load(open(f"{EXPD}/EXPERIMENT_TABLE.json", encoding="utf-8"))
rows = tab["rows"]
BB, SP = tab["bb"], tab["sp"]
sys.path.insert(0, f"{R}/scripts")
import e2e_catalogue as CAT  # noqa: E402

THEMES = {t[0]: (t[1], t[2]) for t in CAT.THEMES}
THEMES["T9"] = ("Fully differentiable builder ladder, successive halving D1-D8 (loop 16, 16b)", "loop 16")
THEMES["T10"] = ("Robustness, memory, map-conditioned builder, consumer under failure (loop 17)", "loop 17")
THEMES["T0"] = ("Runs without a catalogue row", "")
TAGS = {k: v for k, v in CAT.TAGS.items()}
REG = {r["k"]: r for r in json.load(open(f"{R}/literature/register.json", encoding="utf-8"))}

# ------------------------------------------------------------------ extra rows (loop 17 catalogue ids E193 - E212, runs without a row)
L17 = [
    ("E193 / E194", "Champion D7 and the control on the failure series (seven failure types), plus four milder types", "internal names E130, E130b; e17/e130_champion, e130b_more", ["e130_champion", "e130b_more"]),
    ("E195", "Memory on the input: spatial hold, EMA + monitor hold", "internal E131; e17/e131_hold", ["e131_hold"]),
    ("E196", "Builder recursion switched off", "internal E132; e17/e132_norec", ["e132_norec"]),
    ("E197 / E198", "Failure-trained dense head and failure-frame fine-tune of the builder (two folds)", "internal E133a, E133b; e17/e133a_ftall, e133b_k0, e133b_k1, dense_*", ["e133a_ftall", "e133b_k0", "e133b_k1", "dense_e37", "dense_ftall", "dense_trainoof"]),
    ("E199", "Learned gated memory (12.5 k parameters, two folds)", "internal E134; e17/e134_gate", ["e134_gate"]),
    ("E200 / E201", "Route success on the failure series (map assignment under failure)", "internal E135, E135b; e17/e135_route", ["e135_route"]),
    ("E202", "Joint end-to-end fine-tune, three arms (builder + route head + consumer)", "internal E136; e17/e136_*, e17_joint_*", ["e136_graph_a", "e136_graph_b", "e136_graph_base", "e136_graph_c", "e136_route", "@e17_joint_a", "@e17_joint_b", "@e17_joint_c", "@e17_joint_smoke"]),
    ("E203", "Time per frame (latency of the chain under the loop-17 additions)", "internal E137; e17/latency", ["latency"]),
    ("E204 / E205 / E206", "Map-conditioned builder (SD map raster as extra input channels), diagnosis, dense head trained on failures", "internal E138, E138b, E139; e17/e138_map, e139_*", ["e138_map", "e138b_mapdiag.json", "e139_b", "e139_c", "e139_route", "@e17_map_a", "@e17_map_b", "@e17_map_c"]),
    ("E207", "Learned gate behind the map-conditioned builder (frozen builder B, two folds)", "internal E140; e17/e140_*", ["e140_gateB", "e140_route"]),
    ("E208", "Matched clean reference: failure-half frames of the series without perturbation", "internal E141; e17/e141_*", ["e141_clean", "e141_cleanB", "e141_route"]),
    ("E209", "Unseen failure types: builder trained WITHOUT night and blur rows, evaluated on night and blur (rerun pending, the first run crashed)", "internal E142; e17/e142*", []),
    ("E210", "Map-only baseline: the SD map alone drawn as a lane graph", "internal E143; e17/e143_maponly.json", ["e143_maponly.json"]),
    ("E212", "Planner-side consumer under sensor failure", "internal E144; e17/e144_*, consumer", ["e144_champ", "e144_map", "e144_map_m2", "e144_map_none", "e144_test", "consumer"]),
]
extra_rows = [dict(theme="T10", id=i, what=w, result=f"see the loop-17 report and experiment_execution/loop17_recovery ({n})", tag="E", sp=[], bb=[], k=[], _srcs=s) for i, w, n, s in L17]
extra_rows.append(dict(theme="T9", id="E123", what="Differentiable ladder: eight builder designs D1-D8 by successive halving (rungs 0-3), control arms B0-B3", result="champion D7 + junction head + time recursion; see Loop16 report and runs/e123_ladder/TRIAGE_STATE.md", tag="E", sp=["SP5"], bb=[], k=[], _srcs=["@e123_ladder", "@e123"]))
extra_rows.append(dict(theme="T0", id="E34 / E35 / E36", what="Loop 3: map clean / stale / oracle conditions and joint runs (no catalogue row)", result="see reports/source/text (loop 3) and the dense-head runs", tag="C", sp=[], bb=[], k=[], _srcs=["@e34_map_clean", "@e34_map_stale", "@e34_nomap_ft", "@e34_oracle", "@e34_stale_d6", "@e35_A", "@e35_B"]))
for r in extra_rows:
    r["folder"] = folder_name(r)

ALLROWS = rows + extra_rows
for r in rows:
    r.setdefault("folder", folder_name(r))
plan = json.load(open(f"{RS}/plan.json"))
row_of_id = {int(k): v for k, v in plan["row_of_id"].items()}
for r in extra_rows:
    for n in ids_of_row(r["id"]):
        row_of_id[n] = r["folder"] if r["theme"] != "T10" else row_of_id.get(n)
for r in extra_rows:
    if r["theme"] == "T0":
        for n in (34, 35, 36):
            row_of_id[n] = r["folder"]
# the loop-17 catalogue ids collide with loop-16 ledger ids only in the table, not in the run names: run entries named e1NN (NN >= 124) do not exist at top level

LOOPS = [(15, 26, "loop01", "LOOP1"), (27, 43, "loops02-03", None), (44, 56, "loop04", None), (57, 72, "loop05", None), (73, 76, "loop06", None), (77, 80, "loop07", None), (81, 86, "loop08", "LOOP8_INDEX.md"),
         (87, 91, "loop09", None), (92, 104, "loop10", None), (105, 107, "loop11", "LOOP11_INDEX.md"), (108, 112, "loop12", "LOOP12_INDEX.md"), (113, 116, "loop13", "LOOP13_INDEX.md"), (117, 119, "loop14", "LOOP14_INDEX.md"),
         (120, 121, "loop15", "LOOP15_INDEX.md"), (122, 192, "loop16", "TRIAGE_STATE.md"), (193, 212, "loop17", "LOOP17_STATE.md")]


def loop_of(r):
    ns = ids_of_row(r["id"])
    n = ns[0] if ns else 0
    for a, b, nm, idx in LOOPS:
        if a <= n <= b:
            return nm, idx
    return "", None


def locate(rel):
    for base in (RUNS, STG):
        if os.path.lexists(f"{base}/{rel}"):
            return f"{base}/{rel}"
    return None


def gather(src, label):
    """small files below src -> list of (abs, rel under results/figures)"""
    out = []
    if os.path.isfile(src):
        if os.path.splitext(src)[1].lower() in OKEXT and 0 < os.path.getsize(src) <= MAXF:
            out.append((src, label))
        return out
    for dp, dn, fn in os.walk(src):
        for f in fn:
            if os.path.splitext(f)[1].lower() in OKEXT:
                p = os.path.join(dp, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if 0 < sz <= MAXF:
                    out.append((p, label + "/" + os.path.relpath(p, src).replace(os.sep, "/")))
    return out


# ------------------------------------------------------------------ which run entries belong to which folder
srcs = defaultdict(list)  # folder -> [(abs, label)]
for r in extra_rows:
    for s in r.get("_srcs", []):
        if s.startswith("@"):
            nm = s[1:]
            p = locate(nm)
        else:
            nm = s
            p = locate(f"e17/{s}")
        if p:
            srcs[r["folder"]].append((p, nm))
for nm, info in plan["runs"].items():
    if nm.startswith(("e17", "e123")) or info["eid"] is None:
        continue
    eid = info["eid"]
    if eid in row_of_id and eid < 124:
        p = locate(nm)
        if p:
            srcs[row_of_id[eid]].append((p, nm))
# loop-16 ledger rows: cards of the ladder by run id  E123.D6.r0.01 -> e123_D6_r0_01
cards_dir = f"{RUNS}/e123_ladder/cards"
cards = os.listdir(cards_dir) if os.path.isdir(cards_dir) else []
for r in rows:
    m = re.match(r"E123\.([A-Z]\d)\.r(\d)\.([A-Za-z0-9]+)", r["what"])
    if m:
        key = f"e123_{m.group(1)}_r{m.group(2)}_{m.group(3)}"
        for c in cards:
            if c.startswith(key):
                srcs[r["folder"]].append((f"{cards_dir}/{c}", f"cards/{c}"))
        p = locate(f"e123/{key}")
        if p:
            srcs[r["folder"]].append((p, key))
# report figures named e<N>_*.png
figdir = f"{R}/reports/source/figures"
figs = defaultdict(list)
if os.path.isdir(figdir):
    for f in os.listdir(figdir):
        m = re.match(r"e(\d+)[a-z]?_.*\.png$", f)
        if m and int(m.group(1)) in row_of_id and int(m.group(1)) < 124:
            figs[row_of_id[int(m.group(1))]].append(f"{figdir}/{f}")

# ------------------------------------------------------------------ write
index = []
tot_files = tot_bytes = 0
os.makedirs(EXPD, exist_ok=True)
for r in ALLROWS:
    fol = r["folder"]
    d = f"{EXPD}/{fol}"
    files = []
    for p, label in srcs.get(fol, []):
        files += gather(p, label)
    files.sort(key=lambda t: os.path.getsize(t[0]))
    keep, tot = [], 0
    for p, rel in files:
        sz = os.path.getsize(p)
        if tot + sz > MAXTOT:
            continue
        keep.append((p, rel))
        tot += sz
    figl = [(p, os.path.basename(p)) for p in figs.get(fol, [])]
    code = sorted(os.listdir(f"{d}/code")) if os.path.isdir(f"{d}/code") else []
    th = THEMES.get(r["theme"], (r["theme"], ""))
    lp, idx = loop_of(r)
    ks = [k for k in r.get("k", []) if k in REG]
    lines = [f"# {fol}", "",
             f"**{r['what']}**", "",
             "| | |", "|---|---|",
             f"| experiment id | {r['id']} |",
             f"| theme | {r['theme']} - {th[0]} ({th[1]}) |",
             f"| tag | {r['tag']} - {TAGS.get(r['tag'], ('',''))[0]}: {TAGS.get(r['tag'], ('',''))[1]} |" if r["tag"] in TAGS else f"| tag | {r['tag']} |"]
    if r.get("sp"):
        lines.append("| sparsification stage | " + "; ".join(f"{s} {SP.get(s, '')}" for s in r["sp"]) + " |")
    if r.get("bb"):
        lines.append("| building blocks | " + "; ".join(f"{b} {BB.get(b, '')}" for b in r["bb"]) + " |")
    if lp:
        lines.append(f"| loop | {lp}" + (f" - log: [../../experiment_execution/indexes/{idx}](../../experiment_execution/indexes/{idx})" if idx and os.path.exists(f"{R}/experiment_execution/indexes/{idx}") else "") + " |")
    lines += ["", "## Result", "", r["result"], ""]
    if ks:
        lines += ["## Techniques used (register K ids)", ""] + [f"- {k}: {REG[k]['tech'][:140]}  ({REG[k]['status']})" for k in ks] + ["", "Sources and paper notes: [../../literature/register.md](../../literature/register.md)", ""]
    lines += ["## Data", ""]
    if keep:
        top = sorted({rel.split("/")[0] for _, rel in keep})
        lines.append(f"- `results/`: {len(keep)} small files from {', '.join('`' + t + '`' for t in top[:8])}{' ...' if len(top) > 8 else ''}")
    if figl:
        lines.append(f"- `figures/`: {len(figl)} images from the report sources")
    if code:
        lines.append("- `code/`: " + ", ".join(f"`{c}`" for c in code[:14]) + (" ..." if len(code) > 14 else "") + " (reference copy of the scripts that ran it; the shared library is in `../../scripts`)")
    if not (keep or figl or code):
        lines.append("- no separate data files: the numbers above come from the catalogue; see the loop log")
    lines += ["- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`", ""]
    text = "\n".join(lines)
    index.append((fol, r, len(keep), len(figl), len(code)))
    tot_files += len(keep) + len(figl)
    tot_bytes += tot
    if GO:
        os.makedirs(d, exist_ok=True)
        for sub in ("results", "figures"):
            shutil.rmtree(f"{d}/{sub}", ignore_errors=True)
        open(f"{d}/README.md", "w", encoding="utf-8").write(text)
        for p, rel in keep:
            dst = f"{d}/{'figures' if rel.lower().endswith(('.png', '.jpg')) else 'results'}/{rel.replace('/', '__') if rel.lower().endswith(('.png', '.jpg')) else rel}"
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)
        for p, nm in figl:
            os.makedirs(f"{d}/figures", exist_ok=True)
            shutil.copy2(p, f"{d}/figures/{nm}")

# ------------------------------------------------------------------ index
hd = ["# Experiments - one folder per experiment", "",
      f"{len(ALLROWS)} experiments (catalogue ids E15 - E212). Each folder: `README.md` (what, result, links), `results/` (small json / csv / log files), `figures/` (images), `code/` (experiment-specific scripts of finished experiments).",
      "Tags: " + "; ".join(f"**{k}** {v[0]}" for k, v in TAGS.items()) + ".",
      "Folder scheme: `E<first id, three digits><letter>_<slug>`, one folder per catalogue row; rows that cover several ids (e.g. E94b / E96 / E98) carry all of them in the README table and sit under the first. Raw arrays and checkpoints are NOT here (see `../runs`, `../models`).", "",
      "| folder | theme | tag | result | files |", "|---|---|---|---|---|"]
for fol, r, nk, nf, nc in sorted(index, key=lambda t: (ids_of_row(t[1]["id"])[:1] or [999], t[0])):
    res = r["result"].replace("|", "/").replace("\n", " ")
    hd.append(f"| [{fol}]({fol}/README.md) | {r['theme']} | {r['tag']} | {res[:150]}{'...' if len(res) > 150 else ''} | {nk + nf} data, {nc} code |")
print(f"{len(ALLROWS)} experiments; {tot_files} small files, {tot_bytes/1e6:.1f} MB to copy; folders without data: {sum(1 for _, _, a, b, c in index if a + b + c == 0)}")
if GO:
    open(f"{EXPD}/README.md", "w", encoding="utf-8").write("\n".join(hd) + "\n")
    with open(f"{EXPD}/INDEX.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["folder", "ids", "theme", "tag", "what", "result"])
        for fol, r, *_ in index:
            w.writerow([fol, r["id"], r["theme"], r["tag"], r["what"], r["result"]])
