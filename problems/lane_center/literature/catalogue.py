"""Literature catalogue: one row per PAPER, built from the paper notes in literature/papers/.

Usage (plain Python 3, from this folder; research topics are tabled by autoresearch/literature/topic_table.py instead):
    python3 catalogue.py render    # notes + register + papers.csv -> CATALOGUE.md (the one paper reference) and the generated
                                   #   blocks of OVERVIEW.md; refused where code extractions the catalogue lists are missing
                                   #   (--without-code drops their links)
    python3 catalogue.py check     # ids, every relative link, extraction files, verdict, reason and tier codes, port cards
    python3 catalogue.py all       # render + check
    python3 catalogue.py selftest

Files (all in this folder):
    papers/<ID>_<slug>.md   notes, written by agents (level 1: method-section extraction); never edited here
    papers.csv              the ONLY data table, edited by hand or by reviewing agents, one row per paper:
                            id,title,url,axis,tier,verdict,reason,text,idea,extract,code,licence
                            (axis / tier empty = automatic rule; code = repository URL or none; extract = "deep: reason"
                            (no code) or "skip: reason" (neither); title / url / idea only for papers without a note)
    deep/<ID>_<slug>.md     level 2: full-text extraction of a core paper   (found by file name prefix)
    code/<repo>.md          level 3: code extraction of a repository         (found by the repository's name; not in git)
    CATALOGUE.md            GENERATED, the one reference: every paper, relevant first, not relevant last, code table at the end

Model: a PAPER is identified by its first arXiv id (or by the note id if the note has none or is a synthesis
card). Several notes can describe one paper from different angles (cluster of the loop-16 / loop-17 research);
the longest loop-16 note is the home note, the others are listed as "also".
Axes (A1..A9) and tiers (1 core, 2 relevant, 3 background, 4 not suitable) are defined in literature/OVERVIEW.md.
"""
import csv
import os
import re
import shutil
import sys
import tempfile
from urllib.parse import unquote

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))

AXES = {
    "A1": "Sensing foundation (camera + radar BEV)",
    "A2": "Dense lane-centre field",
    "A3": "Sparse lane graph (vertices, edges, junctions)",
    "A4": "Differentiable selection, multi-hypothesis, training regime",
    "A5": "Temporal fusion of detections",
    "A6": "Assignment to a map graph (SD route, lane / road level)",
    "A7": "Navigation context to the planner",
    "A8": "Robustness, sensor failure, uncertainty",
    "A9": "Evaluation, data, metrics",
}
TIERS = {1: "core", 2: "relevant", 3: "background", 4: "not suitable"}
FIELDS = ["key", "axis", "sub", "tier", "home", "notes", "k_ids", "title", "paper", "year", "venue", "arxiv",
          "fit", "stage", "depth", "verified", "code_url", "licence", "deep", "code_extract", "extract",
          "verdict", "reason", "relevance", "idea", "url"]
VERDICTS = {"use", "watch", "context", "not-relevant"}
REASONS = {"lidar", "perspective-only", "trainable-encoder", "hungarian-or-tokens", "aerial-only", "compute", "licence",
           "superseded", "out-of-scope", "no-detail", "other", ""}

# ---------------------------------------------------------------- automatic axis rules
CLUSTER_AXIS = {
    "growth": "A3", "topology": "A3", "verify_sampling": "A3", "vectormap_temporal": "A3",
    "multihyp": "A4", "sensors_bev": "A1", "failure_robustness": "A8",
    "streaming_memory": "A5", "map_matching_planner": "A7", "planner_nav": "A7",
}
SUBS = {
    "growth": "growth agents / tracers", "topology": "topology and linking", "verify_sampling": "readers, verification",
    "vectormap_temporal": "vector-map set predictors", "multihyp": "multi-hypothesis / selection",
}


def auto_axis(cluster, stage):
    """Axis from the cluster and the primary stage code of the home note (overridden by papers.csv)."""
    first = stage[0] if stage else ""
    if cluster == "failure_robustness":
        return "A8"
    if first in ("D", "E"):
        return "A9"
    if first == "S8":
        return "A5"
    if first == "S9":
        return "A6"
    if first == "S10":
        return "A7"
    if first == "B" and cluster in ("sensors_bev", "streaming_memory", "planner_nav", "map_matching_planner"):
        return "A1"
    return CLUSTER_AXIS.get(cluster, "A3")


# ---------------------------------------------------------------- parsing
def parse_stage(text):
    return re.findall(r"\b(S\d+|B|D|L|T|E|X)\b", re.sub(r"\(.*?\)", "", text or ""))


def first(text, pattern, flags=0):
    m = re.search(pattern, text or "", flags)
    return m.group(1).strip() if m else ""


def lead(text, cap=300):
    """First one or two sentences of a note's idea field, at most `cap` characters."""
    t = " ".join((text or "").split())
    if len(t) <= cap:
        return t
    out = ""
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", t):
        if out and len(out) + len(s) > cap:
            break
        out = (out + " " + s).strip()
        if len(out) >= cap * 0.6:
            break
    return out if len(out) <= cap + 80 else out[:cap].rsplit(" ", 1)[0] + " ..."


def parse_note(path):
    txt = read(path)
    fn = os.path.basename(path)
    m = re.match(r"# (\S+) - (.*)", txt)
    nid = m.group(1) if m else fn.split("_")[0]
    title = m.group(2).strip() if m else fn
    d = {"note": nid, "file": fn, "title": title, "len": len(txt)}
    if "\n## Papers" in txt:
        d["fmt"] = 16
        secs = {}
        for s in re.split(r"\n## ", txt)[1:]:
            h, _, body = s.partition("\n")
            secs[h.strip().lower()] = body.strip()
        d["papers"] = secs.get("papers", "")
        d["code"] = secs.get("code", "")
        d["fit"] = secs.get("fit to our project", "")
        d["idea_raw"] = secs.get("idea", "")
        d["stage"] = parse_stage(first(txt, r"Pipeline stage: ([^.]*)\."))
        d["depth"] = first(txt, r"Reading depth: \*\*([^*]+)\*\*")
        d["verified"] = first(txt, r"Verified: ([^.;(]*)")
    else:
        d["fmt"] = 17
        kv = {}
        for line in txt.splitlines():
            mm = re.match(r"- (\w+): (.*)", line)
            if mm:
                kv[mm.group(1)] = mm.group(2)
        d["papers"] = kv.get("papers", "")
        d["code"] = kv.get("code", "")
        d["fit"] = kv.get("fit_to_us", "")
        d["idea_raw"] = kv.get("idea", "")
        d["stage"] = parse_stage(kv.get("stage", ""))
        d["depth"] = kv.get("depth", "")
        d["verified"] = kv.get("verified", "")
    d["cluster"] = first(txt, r"cluster `([^`]+)`")
    m = re.search(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5})", d["papers"])
    d["arxiv"] = m.group(1) if m else ""
    d["idea"] = lead(d.get("idea_raw", ""))
    m2 = re.search(r"https?://[^\s;,)\]]+", d["papers"])
    d["url"] = f"https://arxiv.org/abs/{d['arxiv']}" if d["arxiv"] else (m2.group(0).rstrip(".") if m2 else "")
    synthesis = bool(re.search(r"derived card|synthesis card|\(summary of|^synthesis", title, re.I))
    d["own_key"] = synthesis or not d["arxiv"]
    d["fit_class"] = (re.split(r"[ \-/]", d["fit"].strip())[0] if d["fit"].strip() else "").lower()
    code_blob = d["code"] + " " + d["papers"]
    m = re.search(r"https://github\.com/[\w.\-]+/[\w.\-]+", code_blob)
    d["code_url"] = re.sub(r"\.git$", "", m.group(0)) if m else ""
    # licence: only what the note states within 220 characters after the repository link (most notes state none)
    pos = txt.find(m.group(0)) if m else -1
    win = txt[pos:pos + 220] if pos >= 0 else ""
    lic = first(win, r"(AGPL-3\.0|GPL-3\.0|GPL-2\.0|GPL|LGPL[\w.\-]*|Apache[- ]2\.0|BSD[\w\-]*|CC BY-NC[\w\-]*|CC0|MIT licen[cs]e|licen[cs]e[^;,)]{0,12}MIT)")
    if not lic and re.search(r"no licen[cs]e", win, re.I):
        lic = "none stated"
    d["licence"] = "MIT" if lic.startswith("MIT") or lic.endswith("MIT") else lic.replace("Apache 2.0", "Apache-2.0")
    # first paper entry: "<paper title> (<authors>, <venue> <year>) <url>"
    head = re.split(r"\s+https?://", d["papers"], maxsplit=1)[0]
    mm = re.search(r"^(.*?)\s*\(([^()]*)\)\s*$", head)
    d["paper"] = (mm.group(1) if mm else head).strip()[:160]
    vy = mm.group(2) if mm else ""
    mv = re.search(r"([A-Za-z][\w\-\. /]*?\s*(20\d\d))\s*$", vy)
    d["venue"] = mv.group(1).strip() if mv else ""
    d["year"] = (mv.group(2) if mv else "") or (("20" + d["arxiv"][:2]) if d["arxiv"] else "")
    return d


def read_notes(root):
    pdir = os.path.join(root, "papers")
    out = []
    for fn in sorted(os.listdir(pdir)):
        if fn.endswith(".md") and fn != "INDEX.md":
            out.append(parse_note(os.path.join(pdir, fn)))
    return out


def read_register(root):
    """K id -> {text, note ids, arxiv ids}; used for the K-id column and register-only papers."""
    path = os.path.join(root, "register.md")
    rows = {}
    if not os.path.exists(path):
        return rows
    for line in read(path).splitlines():
        m = re.match(r"\| (K\d+) \| (.*?) \| (.*?) \| (.*?) \| (.*?) \| (.*) \|\s*$", line)
        if not m:
            continue
        k, tech, src, status, used, notes = m.groups()
        rows[k] = {"tech": tech, "source": src, "used": used,
                   "notes": re.findall(r"\(papers/(\w+?)_[^)]*\.md\)", notes),
                   "arxiv": re.findall(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", src)}
    return rows


def read(path):
    """a text file as UTF-8, with or without a byte order mark"""
    with open(path, encoding="utf-8-sig") as f:
        return f.read()


def csv_rows(root):
    """the rows of papers.csv that carry an id, in file order"""
    path = os.path.join(root, "papers.csv")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if (r.get("id") or "").strip()]


def number(text, default):
    """the tier of a papers.csv cell; `default` when it is empty or no number (check names it)"""
    text = (text or "").strip()
    return int(text) if text.isdigit() else default


def read_papers(root):
    """papers.csv: the one data table, edited by hand or by reviewing agents; id -> row (of a double id the last one; check names it).
    Columns: id (note id or arXiv id), title, url, axis, tier, verdict, reason, text, idea, extract, code, licence.
    Rows of papers with a note need only id and the decisions (axis, tier, verdict, reason, text, extract, code, licence);
    title, url and idea are filled in only for papers without a note (found by a search, not yet read)."""
    return {r["id"].strip(): r for r in csv_rows(root)}


# ---------------------------------------------------------------- build
def build(root):
    """one row per paper, sorted by axis, tier and sub-group"""
    notes = read_notes(root)
    reg = read_register(root)
    cur = read_papers(root)
    note_k = {}
    for k, v in reg.items():
        for n in v["notes"]:
            note_k.setdefault(n, []).append(k)
    groups = {}
    for n in notes:
        key = n["note"] if n["own_key"] else n["arxiv"]
        groups.setdefault(key, []).append(n)
    deep_files = sorted(os.listdir(os.path.join(root, "deep"))) if os.path.isdir(os.path.join(root, "deep")) else []
    code_files = os.listdir(os.path.join(root, "code")) if os.path.isdir(os.path.join(root, "code")) else []

    def extractions(ids, urls):
        """(the deep files named after one of the ids, space separated; the code extraction named after one of the repositories)"""
        slugs = {u.rstrip("/").split("/")[-1].lower() + ".md" for u in urls if u.startswith("http")}
        return (" ".join(f for f in deep_files if any(f.startswith(i + "_") for i in ids)),
                next((f for f in sorted(code_files) if f.lower() in slugs), ""))
    rows = []
    for key, g in groups.items():
        g16 = [n for n in g if n["fmt"] == 16] or g
        home = max(g16, key=lambda n: (n["len"], n["note"]))
        ids = sorted(n["note"] for n in g)
        k_ids = sorted({k for i in ids for k in note_k.get(i, [])}, key=lambda s: int(s[1:]))
        fits = [n["fit_class"] for n in g]
        auto_tier = 4 if all(f == "not_suitable" for f in fits) else (
            2 if (k_ids or any(f in ("baseline", "reimplement", "adapt_code", "reuse_code") for f in fits)) else 3)
        axis, tier, code_over, lic_over, ext_over = auto_axis(home["cluster"], home["stage"]), auto_tier, "", "", ""
        for i in [x for x in ids if x != home["note"]] + [home["note"]] + ([key] if key in cur else []):
            c = cur.get(i)  # curation by any note id or by the arXiv key; later entries win, the home note after the others
            if c:
                axis = (c.get("axis") or "").strip() or axis
                tier = number(c.get("tier"), tier)
                code_over = (c.get("code") or "").strip() or code_over
                lic_over = (c.get("licence") or "").strip() or lic_over
                ext_over = (c.get("extract") or "").strip() or ext_over
        no_repo = code_over.lower() == "none"  # papers.csv `code` = none: checked, the paper has no (working) repository
        code_url = "none" if no_repo else (code_over or home["code_url"])  # a URL in papers.csv wins over the note's
        deep, code_x = extractions(ids + [key], (home["code_url"], "" if no_repo else code_over))
        rows.append({
            "key": key, "axis": axis, "sub": SUBS.get(home["cluster"], ""), "tier": tier, "home": home["note"],
            "notes": " ".join(ids), "k_ids": " ".join(k_ids), "title": home["title"], "paper": home["paper"],
            "year": home["year"], "venue": home["venue"], "arxiv": home["arxiv"], "fit": home["fit_class"],
            "stage": ",".join(home["stage"]), "depth": home["depth"], "verified": home["verified"],
            "code_url": code_url, "licence": lic_over or ("" if no_repo else home["licence"]), "deep": deep, "code_extract": code_x,
            "extract": ext_over,
            "verdict": "", "reason": "", "relevance": "", "idea": home["idea"], "url": home["url"],
        })
        rv = next((cur[k] for k in [home["note"], key] + ids if k in cur), None)
        if rv:
            rows[-1].update(verdict=(rv.get("verdict") or "").strip(), reason=(rv.get("reason") or "").strip(),
                            relevance=(rv.get("text") or "").strip())
    # papers that only the register knows (technique register K ids with an arXiv id and no note)
    covered = {r["arxiv"] for r in rows if r["arxiv"]}
    for k, v in sorted(reg.items(), key=lambda kv: int(kv[0][1:])):
        if v["notes"]:
            continue
        for a in v["arxiv"]:
            if a not in covered:
                covered.add(a)
                c = {key: (value or "").strip() for key, value in cur.get(a, {}).items() if key}
                described = c.get("idea") or v["tech"][:300] + " (no note: described only by its register line " + k + ")"
                rows.append({f: "" for f in FIELDS} | {
                    "key": a, "axis": c.get("axis") or "A3", "tier": number(c.get("tier"), 2), "home": "", "notes": "", "k_ids": k,
                    "title": c.get("title") or v["tech"][:110], "arxiv": a, "year": "20" + a[:2], "fit": "register_only", "idea": described,
                    "url": c.get("url") or f"https://arxiv.org/abs/{a}", "code_url": c.get("code", ""), "licence": c.get("licence", ""),
                    "extract": c.get("extract", ""), "verdict": c.get("verdict", ""), "reason": c.get("reason", ""), "relevance": c.get("text", "")})
                rows[-1]["deep"], rows[-1]["code_extract"] = extractions([a, k], (c.get("code", ""),))
    # papers found by a search and not yet read: a row in papers.csv with its own title / url / idea, no note
    known = {r["key"] for r in rows} | {r["home"] for r in rows if r["home"]} | {n for r in rows for n in r["notes"].split()}
    for pid, p in cur.items():
        if pid not in known and (p.get("title") or "").strip():
            a = pid if re.match(r"\d{4}\.\d{4,5}$", pid) else ""
            rows.append({f: "" for f in FIELDS} | {
                "key": pid, "axis": (p.get("axis") or "").strip() or "A3", "tier": number(p.get("tier"), 3),
                "home": "", "notes": "", "title": p["title"].strip(), "arxiv": a, "year": ("20" + a[:2]) if a else "",
                "fit": "found_by_search", "idea": (p.get("idea") or "").strip(),
                "url": (p.get("url") or "").strip() or (f"https://arxiv.org/abs/{a}" if a else ""),
                "code_url": (p.get("code") or "").strip(), "licence": (p.get("licence") or "").strip(),
                "extract": (p.get("extract") or "").strip(), "verdict": (p.get("verdict") or "").strip(),
                "reason": (p.get("reason") or "").strip(), "relevance": (p.get("text") or "").strip()})
            rows[-1]["deep"], rows[-1]["code_extract"] = extractions([pid], (rows[-1]["code_url"],))
    rows.sort(key=lambda r: (r["axis"], r["tier"], r["sub"], r["key"]))
    return rows


# ---------------------------------------------------------------- render
def note_link(root, nid, prefix=""):
    pdir = os.path.join(root, "papers")
    fn = next((f for f in os.listdir(pdir) if f.startswith(nid + "_")), None)
    return f"[{nid}]({prefix}papers/{fn})" if fn else nid


def relink(root, txt):
    """In hand-written files a link `(papers/GR01_anything.md)` is replaced by the real file name of note GR01."""
    pdir = os.path.join(root, "papers")
    files = {f.split("_")[0]: f for f in os.listdir(pdir) if f.endswith(".md") and f != "INDEX.md"}
    return re.sub(r"\(papers/([A-Z]+\d+)(?:_[^)]*)?\.md\)",
                  lambda m: f"(papers/{files[m.group(1)]})" if m.group(1) in files else m.group(0), txt)


def rel_text(r):
    """Novelty / relevance text of a paper; the reason code in front for not-relevant papers."""
    t = (r.get("relevance") or "").strip()
    if (r.get("verdict") or "") == "not-relevant" and r.get("reason"):
        t = f"**not relevant ({r['reason']}).** " + t
    return t.replace("|", "/")


def short_title(title, n=44):
    """Name part of a card title: cut at ' (' / ': ' / ' = ', then at a word boundary."""
    t = re.split(r" \(|: | = ", title)[0].strip()
    if len(t) > n:
        t = t[:n].rsplit(" ", 1)[0] + " ..."
    return t


def short_ref(root, r):
    return f"{r['home'] or r['key']} {short_title(r['title'])}"


def core_line(root, sel):
    """OVERVIEW block: the core papers of an axis in one line; the statements are in CATALOGUE.md."""
    core = [r for r in sel if str(r["tier"]) == "1"]
    if not core:
        return "_no core paper in this axis yet_"
    return f"**Core papers ({len(core)}):** " + "; ".join(short_ref(root, r) for r in core) + "."


def more_line(root, sel):
    """OVERVIEW block: other papers of an axis worth using, and the counts of the rest."""
    others = [r for r in sel if str(r["tier"]) != "1"]
    use = [r for r in others if r.get("verdict") == "use"]
    n = {v: sum(1 for r in others if r.get("verdict") == v) for v in ("watch", "context", "not-relevant")}
    reasons = {}
    for r in others:
        if r.get("verdict") == "not-relevant":
            reasons[r.get("reason") or "other"] = reasons.get(r.get("reason") or "other", 0) + 1
    out = []
    if use:
        out.append("**Also worth using (" + str(len(use)) + "):** " + "; ".join(short_ref(root, r) for r in use) + ".")
    out.append(f"Further: watch {n['watch']}, context {n['context']}, probably not relevant {n['not-relevant']}"
               + (" (" + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])) + ")" if reasons else "")
               + ". Statement and link for every paper: [CATALOGUE.md](CATALOGUE.md).")
    return " ".join(out)


def ref_row(root, r):
    """One table row of the paper reference: id | paper | axis, tier | original | here | content | why."""
    def cell(s):
        return (s or "").replace("|", "/").replace("\n", " ").strip()
    key = r["home"] or r["key"]
    if r["url"]:
        orig = f"[{'arXiv ' + r['arxiv'] if r['arxiv'] else 'page'}]({r['url']})"
    else:
        orig = "no link in the note"
    if r["code_url"].startswith("http"):
        orig += f" [repo]({r['code_url']})"
    here = []
    if r["home"]:
        here.append(note_link(root, r["home"]))
        here += [note_link(root, n) for n in r["notes"].split() if n != r["home"]]
    else:
        here.append(f"register {r['k_ids']} only" if r["k_ids"] else "no note (found by search)")
    here += [f"[deep](deep/{f})" for f in r["deep"].split()]
    if r["code_extract"]:
        here.append(f"[code](code/{r['code_extract']})")
    vy = r["venue"] or r["year"]
    paper = f"{cell(r['title'])} ({cell(vy)})"
    return (f"| **{key}** | {paper} | {r['axis']}, T{r['tier']} | {orig} | {' '.join(here)} | "
            f"{cell(r['idea'])} | {cell(rel_text(r))} |")


GROUPS = [("use", "Relevant: use"), ("watch", "Possibly relevant later: watch"), ("context", "Background context"),
          ("", "Not yet rated"), ("not-relevant", "Probably not relevant (by reason)")]


def write_reference(root, rows):
    """CATALOGUE.md: the one reference file. One row per paper; relevant first, probably-not-relevant last."""
    n_notes = sum(len(r["notes"].split()) for r in rows)
    out = ["# Paper reference - all papers in one list (GENERATED, do not edit)", "",
           f"{len(rows)} papers ({n_notes} notes). One row per paper: what it is, the link to the original, where it lives in this folder, "
           "what it contains, and why it is relevant for our problem or why not. Relevant papers come first, probably-not-relevant "
           "papers last; the code extractions are listed at the end. Goal and the nine axes: [OVERVIEW.md](OVERVIEW.md). "
           "Built by `catalogue.py` (this folder) from the notes and `papers.csv`; change a tier, axis, verdict or statement in `papers.csv`, not here.", "",
           "**Columns.** *id* = note id (or arXiv id for papers only the register knows), grep for it; *axis, T* = axis A1 - A9 and tier "
           "(T1 core, T2 relevant, T3 background, T4 not suitable); *original* = link to the paper; *here* = level-1 note, other notes on the "
           "same paper, `deep` = full-text extraction, `code` = code extraction (not in git); `repo` = the paper's code; *content* = the idea as the note states it; *why* = novelty "
           "and relevance for us in 3 - 4 sentences, for not-relevant papers with the reason in front.", "",
           "| axis | " + " | ".join(v or "unrated" for v, _ in GROUPS) + " | all |", "|---|" + "---|" * (len(GROUPS) + 1)]
    for a, name in AXES.items():
        sel = [r for r in rows if r["axis"] == a]
        out.append(f"| {a} {name} | " + " | ".join(str(sum(1 for r in sel if (r.get("verdict") or "") == v)) for v, _ in GROUPS)
                   + f" | {len(sel)} |")
    out.append("")
    head = "| id | paper | axis, T | original | here | content | why relevant / why not |"
    sep = "|---|---|---|---|---|---|---|"
    for v, title in GROUPS:
        sel = [r for r in rows if (r.get("verdict") or "") == v]
        if not sel and v == "":
            continue
        out += [f"## {title} ({len(sel)})", ""]
        if v == "not-relevant":
            by = {}
            for r in sel:
                by.setdefault(r.get("reason") or "other", []).append(r)
            for reason, lst in sorted(by.items(), key=lambda kv: -len(kv[1])):
                out += [f"### {reason} ({len(lst)})", "", head, sep]
                out += [ref_row(root, r) for r in sorted(lst, key=lambda r: (r["axis"], r["key"]))]
                out.append("")
        else:
            out += [head, sep]
            out += [ref_row(root, r) for r in sorted(sel, key=lambda r: (int(r["tier"]), r["axis"], r["key"]))]
            out.append("")
    out += topic_lines(root)
    out += code_index_lines(root)
    save(os.path.join(root, "CATALOGUE.md"), "\n".join(out))


def save(path, text):
    """write a whole file (Unix line ends) in one step: a failed run leaves the old one"""
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(path + ".tmp", path)


def topic_lines(root):
    """the section of CATALOGUE.md that links the overview of every research topic"""
    base = os.path.join(root, "topics")
    found = []
    for slug in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        path = os.path.join(base, slug, "overview.md")
        if os.path.isfile(path):
            title = read(path).lstrip().split("\n", 1)[0].lstrip("# ").strip() or slug
            found.append(f"- [{slug}](topics/{slug}/overview.md): {title}")
    return ["## Research topics", "", "Each with its own notes, one per paper (procedure: autoresearch/literature).", ""] + (found or ["none yet"]) + [""]


def code_links(text):
    """the code extraction files a catalogue text links"""
    return {f for f in re.findall(r"\]\(code/([^)/]+\.md)\)", text) if not f.startswith(("README", "_"))}


def render(root, without_code=False):
    old = os.path.join(root, "CATALOGUE.md")
    missing = sorted(f for f in code_links(read(old)) if not os.path.exists(os.path.join(root, "code", f))) if os.path.exists(old) else []
    if missing and not without_code:
        raise SystemExit(f"CATALOGUE.md links {len(missing)} code extractions that are not here ({', '.join(missing[:3])}, ...): they are not in "
                         "git; copy them into code/, or render --without-code to drop the links")
    rows = build(root)
    write_reference(root, rows)
    ov = os.path.join(root, "OVERVIEW.md")
    if os.path.exists(ov):
        txt = relink(root, read(ov))
        for a in AXES:
            sel = [r for r in rows if r["axis"] == a and str(r["tier"]) == "1"]
            block = core_line(root, sel)
            txt = re.sub(rf"(<!-- BEGIN core:{a} -->\n).*?(\n<!-- END core:{a} -->)",
                         lambda m: m.group(1) + block + m.group(2), txt, flags=re.S)
            more = more_line(root, [r for r in rows if r["axis"] == a])
            txt = re.sub(rf"(<!-- BEGIN more:{a} -->\n).*?(\n<!-- END more:{a} -->)",
                         lambda m: m.group(1) + more + m.group(2), txt, flags=re.S)
        stats = counts(root, rows)
        txt = re.sub(r"(<!-- BEGIN stats -->\n).*?(\n<!-- END stats -->)", lambda m: m.group(1) + stats + m.group(2), txt, flags=re.S)
        save(ov, txt)
    return rows


def counts(root, rows):
    n_notes = sum(len(r["notes"].split()) for r in rows)
    out = [f"{len(rows)} papers from {n_notes} notes (without a note: {sum(1 for r in rows if not r['notes'])}).", "",
           "| axis | core | relevant | background | not suitable | all |", "|---|---|---|---|---|---|"]
    for a, name in AXES.items():
        sel = [r for r in rows if r["axis"] == a]
        c = [sum(1 for r in sel if str(r["tier"]) == str(t)) for t in TIERS]
        out.append(f"| {a} {name} | {c[0]} | {c[1]} | {c[2]} | {c[3]} | {len(sel)} |")
    vc = {v: sum(1 for r in rows if r.get("verdict") == v) for v in ("use", "watch", "context", "not-relevant")}
    unr = sum(1 for r in rows if not r.get("verdict"))
    out += ["", "Verdicts on relevance for our problem: " + ", ".join(f"{v} {n}" for v, n in vc.items()) + f", not yet rated {unr}."]
    core = [r for r in rows if str(r["tier"]) == "1"]
    skipped = [r for r in core if r["extract"].startswith("skip")]
    queued = [r["home"] or r["key"] for r in core if not r["extract"].startswith("skip") and not r["deep"]]
    out += ["", f"Core papers: {len(core)}; deep extraction done for {sum(1 for r in core if r['deep'])}, still queued: "
                f"{', '.join(queued) or 'none'}; not planned: "
                + ("; ".join(f"{r['home'] or r['key']} ({r['extract'].split(':', 1)[1].strip() if ':' in r['extract'] else 'skip'})"
                             for r in skipped) or "none") + ".",
            f"Code extractions: {len(read_port_cards(root))} files in `code/` "
            "(overview at the end of [CATALOGUE.md](CATALOGUE.md))."]
    return "\n".join(out)


CARD_VALUES = {
    "portable": {"pure-torch", "numpy", "mmcv", "cuda-op", "mixed"},
    "reuse": {"read-only", "adapt", "vendored", "metric-only", "idea-only"},
    "effort": {"S", "M", "L", "none"},
    "status": {"read", "ported", "verified"},
}


def read_port_cards(root):
    """code/<repo>.md -> {file, title, card}; the card is the table under `## Port card` (field | value)."""
    cdir = os.path.join(root, "code")
    out = []
    if not os.path.isdir(cdir):
        return out
    for f in sorted(os.listdir(cdir)):
        if not f.endswith(".md") or f.startswith("_") or f == "README.md" or f.endswith(".skeleton.md"):
            continue
        txt = read(os.path.join(cdir, f))
        card = {}
        m = re.search(r"\n## Port card\n(.*?)(?=\n## |\Z)", txt, re.S)
        if m:
            for k, v in re.findall(r"^\| ([\w]+) \| (.*?) \|\s*$", m.group(1), re.M):
                if k != "field":
                    card[k] = v.strip()
        out.append({"file": f, "title": first(txt, r"# (.*?) - code extraction") or f[:-3], "card": card,
                    "skeleton": os.path.exists(os.path.join(cdir, f[:-3] + ".skeleton.md"))})
    return out


def code_index_lines(root):
    """Last section of CATALOGUE.md: one row per code extraction, from the port card at the top of each code file."""
    cards = read_port_cards(root)
    lines = ["## Code extractions", "",
             "From the port card at the top of each `code/<repo>.md` (process: [code/README.md](code/README.md)). "
             "reuse: read-only (GPL / non-commercial / unlicensed: describe, re-implement), adapt (permissive, with attribution), "
             "vendored, metric-only, idea-only. status: read, ported (our code exists), verified (parity test passed).", "",
             "| repo | papers | licence | portable | reuse | effort | status | port targets | files |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in cards:
        k = c["card"]
        papers = ", ".join(note_link(root, n) for n in k.get("papers", "").split()) or ""
        files = f"[code](code/{c['file']})" + (f" [skeleton](code/{c['file'][:-3]}.skeleton.md)" if c["skeleton"] else "")
        if not k:
            lines.append(f"| {c['title']} | | (no port card yet) | | | | | | {files} |")
        else:
            repo = k.get("repo", "")
            name = f"[{c['title']}]({repo})" if repo.startswith("http") else c["title"]
            lines.append(f"| {name} | {papers} | {k.get('licence', '')} | {k.get('portable', '')} | {k.get('reuse', '')} | "
                         f"{k.get('effort', '')} | {k.get('status', '')} | {k.get('port_targets', '')} | {files} |")
    lines += ["", f"{len(cards)} code extractions, {sum(1 for c in cards if c['card'])} with a port card.", ""]
    return lines


# ---------------------------------------------------------------- check
def broken_links(root):
    """every relative markdown link below `root` that points at nothing; code extractions (not in git) and generated topic tables aside"""
    out = []
    for folder, _, files in os.walk(root):
        for f in sorted(files):
            if not f.endswith(".md"):
                continue
            path = os.path.join(folder, f)
            for link in re.findall(r"(?<!\\)\]\(\s*((?![A-Za-z][A-Za-z0-9+.-]*:|#)[^)\s]+)", re.sub(r"`[^`\n]*`", "", read(path))):
                target = unquote(re.split(r"[#?]", link)[0])
                if re.search(r"(^|/)code/(?!README|_)[^/]+\.md$", target) or re.search(r"(^|/)table(_min\d+)?\.(md|csv)$", target):
                    continue
                if target and not os.path.exists(os.path.join(folder, target)):
                    out.append(f"{os.path.relpath(path, root)}: broken link {link}")
    return out


def check(root):
    rows = build(root)
    problems = []
    pdir = os.path.join(root, "papers")
    ids_on_disk = {f.split("_")[0] for f in os.listdir(pdir) if f.endswith(".md") and f != "INDEX.md"}
    ids_in_cat = {n for r in rows for n in r["notes"].split()}
    for i in sorted(ids_on_disk - ids_in_cat):
        problems.append(f"note {i} is not in the catalogue")
    for r in rows:
        if r["axis"] not in AXES:
            problems.append(f"{r['key']}: bad axis {r['axis']}")
        if r["tier"] not in TIERS:
            problems.append(f"{r['key']}: bad tier {r['tier']}")
        if r["extract"] and not r["extract"].startswith(("deep", "skip")):
            problems.append(f"{r['key']}: extract must start with deep or skip")
    keys_ok = {r["key"] for r in rows} | {r["home"] for r in rows if r["home"]} | ids_in_cat
    claimed = {f for r in rows for f in r["deep"].split()}
    ddir = os.path.join(root, "deep")
    problems += [f"deep/{f} belongs to no paper (name it <id>_<slug>.md)" for f in sorted(os.listdir(ddir) if os.path.isdir(ddir) else [])
                 if f.endswith(".md") and f != "README.md" and not f.startswith("_") and f not in claimed]
    table = os.path.join(root, "papers.csv")
    if os.path.exists(table):
        with open(table, encoding="utf-8-sig", newline="") as f:
            problems += [f"papers.csv line {n}: a row without an id" for n, r in enumerate(csv.DictReader(f), 2)
                         if not (r.get("id") or "").strip() and any((v or "").strip() for k, v in r.items() if k)]
    ids = [p["id"].strip() for p in csv_rows(root)]
    problems += [f"papers.csv: the id {i} has several rows" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for k, p in read_papers(root).items():
        if k not in keys_ok:
            problems.append(f"papers.csv: unknown id {k}")
        v, rs, tier = ((p.get(c) or "").strip() for c in ("verdict", "reason", "tier"))
        if v and v not in VERDICTS:
            problems.append(f"papers.csv {k}: bad verdict '{v}'")
        if rs not in REASONS:
            problems.append(f"papers.csv {k}: bad reason '{rs}'")
        if v == "not-relevant" and not rs:
            problems.append(f"papers.csv {k}: not-relevant without reason")
        if tier and number(tier, 0) not in TIERS:
            problems.append(f"papers.csv {k}: bad tier '{tier}'")
    for c in read_port_cards(root):
        for field, ok in CARD_VALUES.items():
            v = c["card"].get(field)
            if v is not None and (v.split() or [""])[0].strip("`") not in ok and v.strip("`") not in ok:
                problems.append(f"code/{c['file']}: port card {field} = '{v}' not in {sorted(ok)}")
    problems += [f"{fn} missing" for fn in ("README.md", "OVERVIEW.md", "CATALOGUE.md") if not os.path.exists(os.path.join(root, fn))]
    return problems + broken_links(root)


# ---------------------------------------------------------------- selftest
def selftest():
    d = tempfile.mkdtemp(prefix="catalogue_selftest_")
    try:
        _selftest(d)
    finally:
        shutil.rmtree(d)
    print("selftest ok")


def _selftest(d):
    os.makedirs(os.path.join(d, "papers"))
    a16 = ("# GR99 - Alpha (growth agent)\n\n*Pass: loop 16, cluster `growth` (x). Reading depth: **M+C** (A). Verified: checked (y). "
           "Pipeline stage: S4, S1.*\n\n## Papers\n\nAlpha: A test (Xu et al., ICRA 2023) https://arxiv.org/abs/2209.00001\n\n## Idea\n\n"
           "Alpha grows a graph. It is new.\n\n## Code\n\nhttps://github.com/foo/alpha ; GPL-3.0\n\n## Fit to our project\n\nbaseline - effort M\n")
    b17 = ("# A99 - Alpha again (streaming)\n\n*Pass: loop 17*\n\n- papers: Alpha: A test (Xu et al., ICRA 2023) https://arxiv.org/abs/2209.00001\n"
           "- stage: S8\n- fit_to_us: reuse_idea - S\n- depth: M\n- verified: checked\n")
    c = ("# SN98 - Derived card\n\n*Pass: loop 16, cluster `sensors_bev` (x). Reading depth: **M** (A). Verified: checked. Pipeline stage: D.*\n\n"
         "## Papers\n\nsomething https://arxiv.org/abs/2209.00001\n\n## Fit to our project\n\nnot_suitable\n")
    for fn, t in (("GR99_alpha.md", a16), ("A99_alpha-again.md", b17), ("SN98_derived.md", c)):
        open(os.path.join(d, "papers", fn), "w", encoding="utf-8").write(t)
    open(os.path.join(d, "papers.csv"), "w", encoding="utf-8").write(
        "id,title,url,axis,tier,verdict,reason,text,idea,extract,code,licence\n"
        'GR99,,,A5,1,use,,"New thing. Touches the builder. Blocker none. Use it.",,"deep: no code wanted",,MIT\n'
        'SN98,,,,,not-relevant,lidar,"Needs lidar.",,,none,\n'
        '2611.00001,Beta found by search,,A6,2,watch,,"Maybe later.",Beta does a thing.,,https://github.com/foo/beta,\n'
        '2612.00002,Gamma the paper,,,,watch,,"Later.",Gamma does another thing.,,https://github.com/foo/gamma,\n')
    open(os.path.join(d, "register.md"), "w", encoding="utf-8").write(
        "| id | technique | source | status | used | notes |\n|---|---|---|---|---|---|\n"
        "| K1 | a gamma loss | https://arxiv.org/abs/2612.00002 | idea | - | - |\n")
    rows = build(d)
    assert len(rows) == 4, rows          # GR99 + A99 share arXiv 2209.00001, SN98 is a derived card, Beta has no note, Gamma is in the register
    g = [x for x in rows if x["key"] == "2612.00002"][0]   # a register paper keeps the title, idea and repository of its papers.csv row
    assert (g["title"], g["idea"], g["code_url"], g["k_ids"]) == ("Gamma the paper", "Gamma does another thing.", "https://github.com/foo/gamma", "K1"), g
    r = [x for x in rows if x["key"] == "2209.00001"][0]
    assert r["home"] == "GR99" and r["notes"] == "A99 GR99" and r["verdict"] == "use", r
    assert r["axis"] == "A5" and r["tier"] == 1 and r["extract"] == "deep: no code wanted", r
    assert r["code_url"] == "https://github.com/foo/alpha" and r["licence"] == "MIT", r
    assert r["year"] == "2023" and r["venue"] == "ICRA 2023" and r["idea"].startswith("Alpha grows"), r
    s = [x for x in rows if x["key"] == "SN98"][0]
    assert s["axis"] == "A9" and s["tier"] == 4 and s["code_url"] == "none" and s["reason"] == "lidar", s
    b = [x for x in rows if x["key"] == "2611.00001"][0]
    assert b["title"] == "Beta found by search" and b["url"].endswith("2611.00001") and b["axis"] == "A6", b
    assert auto_axis("planner_nav", ["S9"]) == "A6" and auto_axis("multihyp", ["L"]) == "A4"
    os.makedirs(os.path.join(d, "code"))
    open(os.path.join(d, "code", "alpha_road.md"), "w", encoding="utf-8").write("# alpha_road - code extraction\n")
    assert [x for x in build(d) if x["key"] == "2209.00001"][0]["code_extract"] == "", "a name that only starts like the repository is not it"
    open(os.path.join(d, "code", "alpha.md"), "w", encoding="utf-8").write(
        "# alpha - code extraction\n\n## Port card\n\n| field | value |\n|---|---|\n| repo | https://github.com/foo/alpha |\n"
        "| licence | MIT |\n| reuse | adapt |\n| effort | S |\n| status | read |\n| papers | GR99 |\n\n## 1. Map of the code\n")
    cards = read_port_cards(d)
    card = [c for c in cards if c["file"] == "alpha.md"][0]["card"]
    assert len(cards) == 2 and card["reuse"] == "adapt" and card["papers"] == "GR99", cards
    assert any("| adapt |" in line for line in code_index_lines(d))
    for name, text in (("code/README.md", "# c\n"), ("README.md", "# r\n"), ("OVERVIEW.md", "# o\n<!-- BEGIN stats -->\n\n<!-- END stats -->\n")):
        open(os.path.join(d, name), "w", encoding="utf-8").write(text)
    for name in ("deep", "topics/t1"):
        os.makedirs(os.path.join(d, name))
    open(os.path.join(d, "deep", "2612.00002_gamma.md"), "w", encoding="utf-8").write("# deep\n")
    open(os.path.join(d, "topics", "t1", "overview.md"), "w", encoding="utf-8").write("# Overview - a topic\n`W[c](x)`\n")
    open(os.path.join(d, "deep", "2612.00002_gamma_again.md"), "w", encoding="utf-8").write("# deep\n")
    assert [x for x in build(d) if x["key"] == "2612.00002"][0]["deep"] == "2612.00002_gamma.md 2612.00002_gamma_again.md", "every deep file of a paper"
    open(os.path.join(d, "code", "beta.md"), "w", encoding="utf-8").write("# beta - code extraction\n\n## Port card\n\n| field | value |\n|---|---|\n| reuse |  |\n")
    render(d)
    catalogue = read(os.path.join(d, "CATALOGUE.md"))
    assert "[t1](topics/t1/overview.md): Overview - a topic" in catalogue, catalogue
    assert "[repo](https://github.com/foo/alpha)" in catalogue and "[code](code/alpha.md)" in catalogue, catalogue
    assert "Code extractions: 3 files" in read(os.path.join(d, "OVERVIEW.md")), "the count is of this folder"
    assert check(d) == ["code/beta.md: port card reuse = '' not in " + str(sorted(CARD_VALUES["reuse"]))], check(d)
    os.remove(os.path.join(d, "code", "beta.md"))
    os.rename(os.path.join(d, "code", "alpha.md"), os.path.join(d, "alpha.away"))
    try:
        render(d)
        raise AssertionError("a render without a listed code extraction")
    except SystemExit as e:
        assert "alpha.md" in str(e), e
    assert "[code](code/alpha.md)" in read(os.path.join(d, "CATALOGUE.md")), "nothing was written"
    render(d, without_code=True)
    assert "code/alpha.md" not in read(os.path.join(d, "CATALOGUE.md"))
    os.rename(os.path.join(d, "alpha.away"), os.path.join(d, "code", "alpha.md"))
    render(d)
    assert check(d) == [], check(d)
    open(os.path.join(d, "deep", "stray_notes.md"), "w", encoding="utf-8").write("# s\n")
    open(os.path.join(d, "papers.csv"), "a", encoding="utf-8").write(',Lost,https://x.org/lost,,,,,,,,,\n')
    found = check(d)
    assert any("stray_notes.md belongs to no paper" in p for p in found) and any("a row without an id" in p for p in found), found
    os.remove(os.path.join(d, "deep", "stray_notes.md"))
    open(os.path.join(d, "papers", "GR99_alpha.md"), "a", encoding="utf-8").write("\n[gone](../nowhere.md) [ok](GR99_alpha.md) [x](code/other.md)\n")
    open(os.path.join(d, "papers.csv"), "a", encoding="utf-8").write('GR99,,,A5,high,use,,"again",,,,\n')
    found = check(d)
    assert any("broken link ../nowhere.md" in p for p in found) and not any("other.md" in p for p in found), found
    assert any("GR99 has several rows" in p for p in found) and any("bad tier 'high'" in p for p in found), found


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--without-code"]
    cmd = args[0] if args else "all"
    root = os.path.abspath(args[1]) if len(args) > 1 else DEFAULT_ROOT
    if cmd not in ("render", "check", "all", "selftest") or len(args) > 2:
        print("usage: catalogue.py [render | check | all | selftest] [literature folder] [--without-code]", file=sys.stderr)
        sys.exit(2)
    if cmd == "selftest":
        selftest()
        sys.exit(0)
    if cmd in ("render", "all"):
        render(root, "--without-code" in sys.argv)
        print(f"render: {len(build(root))} papers -> CATALOGUE.md and the generated blocks of OVERVIEW.md")
    if cmd in ("check", "all"):
        probs = check(root)
        print("check:", "ok" if not probs else f"{len(probs)} problems")
        for p in probs[:60]:
            print("  ", p)
        if probs:
            sys.exit(1)
