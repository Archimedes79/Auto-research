"""Build C:/ML-Experiment/literature from the literature work of the loops:
papers/<id>_<slug>.md  one note per card (301 cards of the loop-16 pass, 103 cards of the loop-17 pass)
papers/INDEX.md        table of all papers
repos/README.md        repositories named in the cards + the three local clones (read-only look at git metadata)
register.md            K register (techniques with sources), linked to the paper notes
sources/               the original research notes (copies)
Nothing is downloaded. Run in WSL: _py14.sh ../restructure/build_literature.py"""
import csv
import json
import os
import re
import shutil
import subprocess
from collections import defaultdict

ROOT = "/mnt/c/ML-Experiment"
RF = f"{ROOT}/report_final"
TXT = f"{RF}/text"
L = f"{ROOT}/literature"
for d in ("papers", "repos", "sources"):
    os.makedirs(f"{L}/{d}", exist_ok=True)

CLUSTER = {
    "GR": ("growth", "iterative growth of road / lane graphs (agents, walkers, trackers)"),
    "TP": ("topology", "lane topology heads, link scoring, graph priors"),
    "VM": ("vectormap_temporal", "online vector maps and temporal / streaming fusion"),
    "VS": ("verify_sampling", "verification along rays, sampling, survival and hazard read-outs"),
    "MH": ("multihyp", "multiple hypotheses, set prediction, matching-free losses"),
    "SN": ("sensors_bev", "camera-radar BEV foundations, sensors, data sets, robustness"),
    "PL": ("planner_nav", "planner interface, navigation, map matching, end-to-end driving"),
    "A": ("streaming_memory", "loop 17: streaming / temporal memory for online maps"),
    "B": ("failure_robustness", "loop 17: sensor failure, robustness, gating"),
    "C": ("map_matching_planner", "loop 17: SD-map matching and the planner consumer"),
}


def slug(s, n=5):
    s = re.sub(r"\(.*?\)", " ", s)
    w = re.findall(r"[A-Za-z0-9]+", s.lower())
    return "-".join(w[:n]) or "card"


URL = re.compile(r"https?://[^\s;,)\]>\"']+")
GH = re.compile(r"https?://github\.com/[A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+")
ARX = re.compile(r"(?:arxiv\.org/(?:abs|pdf|html)/|arXiv[: ]\s?)(\d{4}\.\d{4,5})", re.I)
LIC = re.compile(r"(Apache[- ]?2\.0|MIT licen[cs]e|\bMIT\b|GPL[- ]?v?\d?|AGPL[- ]?\d?|BSD[-\w]*|CC[- ]BY[-\w. ]*|CC BY-NC[-\w. ]*|no licen[cs]e[^.;]*|licen[cs]e not [a-z]+|NVIDIA[^.;]*licen[cs]e)", re.I)

papers = []  # dict(id, name, cluster, file, links, arx, depth, verified, fit, repos, code)

# ---------------------------------------------------------------- loop-16 cards (cards.csv)
rows = list(csv.DictReader(open(f"{TXT}/lit16/cards.csv", encoding="utf-8", newline="")))
FIELDS = [("idea", "Idea"), ("formulation", "Formulation"), ("input", "Input"), ("output", "Output"), ("training", "Training"), ("differentiable", "Differentiable?"),
          ("sensors_data", "Sensors and data"), ("results", "Reported results"), ("compute", "Compute"), ("code", "Code"), ("fit_to_us", "Fit to our project"),
          ("compat_needs", "Needs"), ("compat_conflicts", "Conflicts"), ("tried_before", "Tried in our experiments")]
for r in rows:
    pre = re.match(r"[A-Z]+", r["id"]).group(0)
    fn = f"{r['id']}_{slug(r['name'])}.md"
    links = list(dict.fromkeys(u.rstrip(".") for u in URL.findall(r["papers"])))
    out = [f"# {r['id']} - {r['name']}", "",
           f"*Pass: loop 16 (2026-10-23), cluster `{r['cluster']}` ({CLUSTER[pre][1]}). Reading depth: **{r['depth']}** (A abstract, M method, C code page). Verified: {r['verified']}. Pipeline stage: {r['stage']}.*", "",
           "## Papers", "", r["papers"].strip(), ""]
    for k, h in FIELDS:
        v = r[k].strip()
        if v:
            out += [f"## {h}", "", v, ""]
    open(f"{L}/papers/{fn}", "w", encoding="utf-8").write("\n".join(out))
    papers.append(dict(id=r["id"], name=r["name"], cluster=r["cluster"], file=fn, links=links, arx=ARX.findall(r["papers"]), depth=r["depth"], verified=r["verified"].strip(" ."), fit=r["fit_to_us"].split(" ")[0][:40], code=r["code"].strip(), tried=r["tried_before"].strip(),
                       repos=list(dict.fromkeys(u.rstrip(".") for u in GH.findall(r["code"] + " " + r["papers"])))))

# ---------------------------------------------------------------- loop-17 cards (A / B / C markdown)
for pre, fname in (("A", "A_streaming_memory.md"), ("B", "B_failure_robustness.md"), ("C", "C_map_matching_planner.md")):
    txt = open(f"{TXT}/lit17/{fname}", encoding="utf-8").read()
    parts = re.split(r"^### CARD ", txt, flags=re.M)
    for part in parts[1:]:
        head, _, body = part.partition("\n")
        m = re.match(r"([A-Z]+\d+) - (.*)", head)
        if not m:
            continue
        cid, name = m.group(1), m.group(2).strip()
        body = re.split(r"^## ", body, maxsplit=1, flags=re.M)[0]  # stop at the next top-level section of the file
        fn = f"{cid}_{slug(name)}.md"
        links = list(dict.fromkeys(u.rstrip(".") for u in URL.findall(body)))
        depth = (re.search(r"[Dd]epth[^A-Za-z0-9]*:?\**\s*`?([AMC](?:\+[AMC])?)", body) or [None, "?"])[1]
        open(f"{L}/papers/{fn}", "w", encoding="utf-8").write(f"# {cid} - {name}\n\n*Pass: loop 17 (2026-10-24), file `{fname}`, cluster `{CLUSTER[pre][0]}`. Depth M here means: method sections queried through a page digest (see sources/lit17/{fname}, section 1).*\n\n" + body.strip() + "\n")
        papers.append(dict(id=cid, name=name, cluster=CLUSTER[pre][0], file=fn, links=links, arx=ARX.findall(body), depth=depth, verified="checked (digest)", fit="", code="", tried="",
                           repos=list(dict.fromkeys(u.rstrip(".") for u in GH.findall(body)))))

# ---------------------------------------------------------------- paper index
by_cluster = defaultdict(list)
for p in papers:
    by_cluster[p["cluster"]].append(p)
idx = ["# Paper notes - index", "", f"{len(papers)} notes, one file per technique / paper family (a card may cover several papers of one family). Links go to the paper pages; the note holds the summary written from the method sections.", "",
       "Reading depth: A = abstract page, M = method section, C = code page. Details of how each note was produced: `../sources/lit16/SCHEMA.md` (loop 16) and the coverage statement in `../sources/lit17/*.md` (loop 17).", ""]
for pre, (cl, desc) in CLUSTER.items():
    ps = by_cluster.get(cl, [])
    if not ps:
        continue
    idx += [f"## {cl} - {desc} ({len(ps)})", "", "| id | technique | depth | verified | paper link |", "|---|---|---|---|---|"]
    for p in ps:
        lk = f"[link]({p['links'][0]})" if p["links"] else ""
        idx.append(f"| [{p['id']}]({p['file']}) | {p['name'][:110].replace('|', '/')} | {p['depth']} | {p['verified'][:18]} | {lk} |")
    idx.append("")
open(f"{L}/papers/INDEX.md", "w", encoding="utf-8").write("\n".join(idx))
arx_to_file = {}
for p in papers:
    for a in p["arx"]:
        arx_to_file.setdefault(a, (p["id"], p["file"]))

# ---------------------------------------------------------------- register K
tab = json.load(open(f"{ROOT}/out/loop2/EXPERIMENT_TABLE.json", encoding="utf-8"))
reg = ["# Technique register (K ids)", "", f"{len(tab['register'])} techniques that were registered with source and status before they were used in an experiment (rule of the loops: no technique without a K id). `status` says whether the source page was seen (checked) or the entry is from memory / an own idea. The last column links to a paper note when the source names an arXiv id that a note covers.", "",
       "| K | technique | source | status | used in | paper note |", "|---|---|---|---|---|---|"]
REG = tab["register"]
REG = [dict(k=k, **v) for k, v in REG.items()] if isinstance(REG, dict) else REG
for r in REG:
    notes = []
    for a in dict.fromkeys(ARX.findall(r.get("src", "") + " " + r.get("tech", ""))):
        if a in arx_to_file:
            notes.append(f"[{arx_to_file[a][0]}](papers/{arx_to_file[a][1]})")
    c = lambda s: str(s).replace("|", "/").replace("\n", " ")
    reg.append(f"| {r['k']} | {c(r.get('tech',''))} | {c(r.get('src',''))} | {c(r.get('status',''))} | {c(r.get('used',''))} | {', '.join(notes)} |")
open(f"{L}/register.md", "w", encoding="utf-8").write("\n".join(reg) + "\n")
json.dump(REG, open(f"{L}/register.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)

# ---------------------------------------------------------------- repositories
repo_cards = defaultdict(list)
repo_note = {}
for p in papers:
    for u in p["repos"]:
        key = u.lower()
        repo_cards[key].append(p)
        repo_note.setdefault(key, (u, p["code"]))
lines = ["# Repositories", "", "Two parts: (1) the code bases that are cloned in the WSL home and used by the pipeline, (2) every repository named in the paper notes (cards). Nothing here was cloned for the notes; licences are as written on the cards (`not checked` = nobody looked).", "",
         "## 1. Local clones (WSL `~/work`)", ""]
LOCAL = [("CRN", "CRN (Camera Radar Net), ICCV 2023", "Detection-pretrained camera-radar foundation of loops 1-3 (E15-E26); replaced by GaussianCaR from E27."),
         ("gaussiancar", "GaussianCaR, ICRA 2026", "Frozen map-pretrained camera-radar foundation of all later loops (39 channels, 200 x 200 cells of 0.5 m)."),
         ("bevfusion", "BEVFusion (MIT HAN Lab)", "Camera-lidar strand E01-E14, results deleted 2026-10-13; code kept."),
         ("gcar_probe", "gcar probe scripts", "First trial runs of the GaussianCaR environment (own scripts, not a clone).")]
home = os.path.expanduser("~/work")
for d, title, use in LOCAL:
    p = f"{home}/{d}"
    info = []
    if os.path.isdir(f"{p}/.git"):
        def g(*a):
            return subprocess.run(["git", "-C", p, *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout.strip()
        rem = g("remote", "get-url", "origin")
        last = g("log", "-1", "--format=%h %ad %s", "--date=short")
        info = [f"- remote: {rem}" if rem else "- remote: none", f"- last commit: {last}"]
    lic = [f for f in os.listdir(p) if f.lower().startswith("licen")] if os.path.isdir(p) else []
    lines += [f"### {title}", "", f"- local path: `~/work/{d}` ({'present' if os.path.isdir(p) else 'missing'})", *info, f"- licence file: {', '.join(lic) if lic else 'none found at top level'}", f"- used for: {use}", ""]
lines += ["Weights: `~/work/gaussiancar_ckpt/nuscenes_map.ckpt` (GaussianCaR map-segmentation checkpoint).", "", f"## 2. Repositories named in the paper notes ({len(repo_cards)})", "", "| repository | licence (as on the card) | notes that use it |", "|---|---|---|"]
for key in sorted(repo_cards):
    u, code = repo_note[key]
    m = LIC.search(code or "")
    lic = m.group(0) if m else ("not checked" if not code else "see card")
    ids = ", ".join(f"[{p['id']}](../papers/{p['file']})" for p in repo_cards[key][:6])
    lines.append(f"| [{u.replace('https://github.com/', '')}]({u}) | {lic[:40]} | {ids} |")
open(f"{L}/repos/README.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- original research notes
SRC = [(f"{TXT}/lit_condensed.md", "lit_condensed.md"), (f"{TXT}/proposal_sparse.md", "proposal_sparse.md")]
for f in sorted(os.listdir(TXT)):
    if f.startswith(("lit_extra", "lit_p16")) and f.endswith(".md"):
        SRC.append((f"{TXT}/{f}", f))
for d in ("lit16", "lit17"):
    for f in sorted(os.listdir(f"{TXT}/{d}")):
        if os.path.isfile(f"{TXT}/{d}/{f}") and not f.endswith(".bak6") and "lit_" not in f:
            SRC.append((f"{TXT}/{d}/{f}", f"{d}/{f}"))
for d in ("lit7", "lit8"):
    for f in sorted(os.listdir(f"{RF}/{d}")):
        SRC.append((f"{RF}/{d}/{f}", f"{d}/{f}"))
n = 0
for s, name in SRC:
    if not os.path.isfile(s):
        continue
    o = f"{L}/sources/{name}"
    os.makedirs(os.path.dirname(o), exist_ok=True)
    shutil.copy2(s, o)
    n += 1
print("papers:", len(papers), "repos:", len(repo_cards), "register:", len(tab["register"]), "source files copied:", n)
print({c: len(v) for c, v in by_cluster.items()})
json.dump(papers, open(f"{ROOT}/restructure/papers.json", "w"), indent=0)
