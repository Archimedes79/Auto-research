#!/usr/bin/env python3
"""reporter: build a paper-style, engineer-readable report (one html file + media) from markdown parts, fill the numbers from result
files, check every number against its sources, list the kept helper scripts, and pack report + code + utilities into a size-limited zip.

    python reporter.py new <folder> [--title T]      scaffold a report folder (report.json + the paper chapters, each with a prompt)
    python reporter.py build <folder> [--out F.html] [--strict]
    python reporter.py check <folder> [--strict]     every decimal number of the text must be found in the sources
    python reporter.py pack  <folder> [--out F.zip]  html + media + read-me + code + utilities, refuses over the size limit
    python reporter.py selftest                      builds, checks and packs a throw-away example report

Needs: python 3.8+, markdown-it-py; Pillow is optional (recompresses images). Standalone: no import from any project.
Full description with a minimal example: README.md. The procedure and the quality bar for writing the text: SKILL.md."""
import argparse, base64, functools, glob, io, json, os, re, shutil, subprocess, sys, tempfile, zipfile

PENDING = "PENDING"
PLACEHOLDER_SVG = "PENDING: architecture drawing"                       # the text of the scaffold's placeholder drawing (`new`); `drawings` does not count it
NUM = re.compile(r"(?<![\w.])-?\d+\.\d+(?![\w.])")
CHAPTERS = [  # id, title, part file, prompt written into the scaffold (what the chapter must contain)
    ("intro", "1. Introduction: purpose and status", "01_intro.md",
     "PENDING: the problem in plain words (three lines), what was built, the status in one line (what works, what does not), "
     "the headline result with its baseline, and a map of the report. The abstract and the teaser figure (report.json) come before this chapter."),
    ("system", "2. The system: architecture and description", "02_system.md",
     "PENDING: one drawing (parts/02_system.svg, the same colours for the same kinds of block), the chain in a few lines, what is learned and what is a named "
     "rule or budget, the loss in one paragraph, parameters and time per frame. Define the terms once."),
    ("gt", "3. Required ground truth", "03_gt.md",
     "PENDING: what the system needs as labels or reference and why, how it is made (with pictures of real examples: the raw data next to the derived labels), "
     "how to know it is right, what it costs and where it is limited."),
    ("exp", "4. Experiments done, very short", "04_experiments.md",
     "PENDING: one row per step: the idea in one line, the result, the verdict. Negative results stay in. Table: caption on the line before the table."),
    ("quant", "5. Quantitative results", "05_quant.md",
     "PENDING: tables with the split and the denominator named; controls and baselines next to the result; seeds and noise band; plots, images and videos from parts/ and media/."),
    ("qual", "6. Qualitative results: good cases and issues", "06_qual.md",
     "PENDING: real examples chosen by a written rule, good cases first, then the failure modes (each with a count of how often it happens); the same input with "
     "the baseline and the winner side by side; every caption says what to look at."),
    ("winners", "7. The winning systems", "07_winners.md",
     "PENDING: for each winner: its architecture (a drawing or a short chain), the headline numbers, parameters, time per frame, and where the weights and the run folder are."),
    ("outlook", "8. Conclusion and next steps", "08_outlook.md",
     "PENDING: what the work shows in three lines; what is still hard; the next steps, very brief, with what each costs; decisions that wait for the owner."),
    ("repeat", "A. How to repeat the result", "A1_repeat.md",
     "PENDING: data and versions, the commands in order, seeds, the numbers to hit at each stage, the checks that tell a rebuild is right, known pitfalls."),
]
CSS = """
:root{--bg:#fdfcf9;--fg:#1d2330;--mut:#5b6475;--ln:#d9d6cc;--ac:#1f5fa8;--bgc:#f3f1ea;--pe:#fff1c9;--pf:#7a5a00}
@media (prefers-color-scheme:dark){:root{--bg:#161a22;--fg:#e6e8ee;--mut:#9aa3b5;--ln:#2d3443;--ac:#6aa7ee;--bgc:#1e2330;--pe:#4a3d12;--pf:#ffe08a}}
body{background:var(--bg);color:var(--fg);font:17px/1.62 Georgia,"Iowan Old Style","Palatino Linotype",serif;margin:0}
main{max-width:860px;margin:0 auto;padding:28px 18px 80px}
h1,h2,h3,h4,nav,figcaption,table,.sub,.abs b.lab{font-family:system-ui,"Segoe UI",Arial,sans-serif}
h1{font-size:30px;line-height:1.2;margin:.1em 0 .2em;letter-spacing:-.01em}
h2{font-size:21px;margin:2em 0 .5em;border-bottom:1px solid var(--ln);padding-bottom:5px}
h3{font-size:17px;margin:1.4em 0 .3em}h4{font-size:15px;margin:1em 0 .2em}
code{background:var(--bgc);padding:1px 4px;border-radius:3px;font-size:.84em;font-family:Consolas,monospace}pre{background:var(--bgc);padding:9px;overflow-x:auto;font-size:.8em}
nav{font-size:13px;color:var(--mut);margin:10px 0 18px;line-height:1.9}nav a{color:var(--ac);text-decoration:none;margin-right:10px;white-space:nowrap}
.sub{color:var(--mut);font-size:13.5px}
.abs{background:var(--bgc);border-left:4px solid var(--ac);padding:10px 16px;margin:16px 0;font-size:15.5px}.abs b.lab{display:block;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);margin-bottom:3px}
.abs p{margin:.4em 0;max-width:none}
figure{margin:1.3em 0;text-align:center}figure img,figure video,figure svg{max-width:100%;height:auto;border:1px solid var(--ln);background:#fff}
figcaption{font-size:13px;color:var(--mut);text-align:left;margin-top:6px;line-height:1.45}figcaption b{color:var(--fg)}
.tw{overflow-x:auto;margin:1em 0}table{border-collapse:collapse;font-size:12.5px;margin:0 auto}th,td{border:1px solid var(--ln);padding:3px 7px;vertical-align:top}th{background:var(--bgc);text-align:left}
caption{caption-side:top;text-align:left;font-size:13px;color:var(--mut);padding:0 0 5px;line-height:1.45}caption b{color:var(--fg)}
p,li{max-width:72ch}
.pend{background:var(--pe);color:var(--pf);font-weight:700;padding:0 4px;border-radius:3px;font-size:.82em;font-family:system-ui,sans-serif}
@media print{nav{display:none}body{font-size:11.5pt;background:#fff;color:#000}main{max-width:none;padding:0}figure,table{break-inside:avoid}}
"""


# ------------------------------------------------------------------ helpers
def load_spec(folder):
    p = os.path.join(folder, "report.json")
    if not os.path.exists(p):
        sys.exit("no report.json in %s (python reporter.py new <folder>)" % folder)
    return json.load(open(p, encoding="utf-8"))


def locate(folder, spec, path, sub=""):
    """the file a path of report.json or of a part names; relative to the report folder (sub = "parts": a section's part and svg are relative to parts/). A path that
    leaves the report folder is read from spec["origin"] when that is given: the folder this one is a copy of (`research report` builds a copy outside git)"""
    if os.path.isabs(path):
        return path
    p = os.path.normpath(os.path.join(sub, path))
    outside = p == os.pardir or p.startswith(os.pardir + os.sep)
    return os.path.normpath(os.path.join((outside and spec.get("origin")) or folder, p))


def dig(obj, key):
    """'a.b.0.c' into nested dicts / lists; a dict key may hold dots itself ('numbers.graph.topo_F1' of a result or champion record): at each
    level the longest key that matches wins"""
    parts = key.split(".")
    while parts:
        if isinstance(obj, list):
            obj, parts = obj[int(parts[0])], parts[1:]
            continue
        n = next((n for n in range(len(parts), 0, -1) if ".".join(parts[:n]) in obj), 0)
        if not n:
            raise KeyError(key)
        obj, parts = obj[".".join(parts[:n])], parts[n:]
    return obj


def resolve_values(folder, spec):
    """name -> text. A value is a literal string / number, or {"file": path, "key": "a.b.0", "fmt": "{:.4f}"}; a missing file or key gives PENDING."""
    out, missing = {}, []
    for name, v in spec.get("values", {}).items():
        if isinstance(v, dict) and "file" in v:
            try:
                x = json.load(open(locate(folder, spec, v["file"]), encoding="utf-8"))
                x = dig(x, v["key"]) if v.get("key") else x
                out[name] = v.get("fmt", "{}").format(x)
            except Exception:
                out[name] = PENDING
                missing.append(name)
        else:
            out[name] = str(v)
    return out, missing


def fill(text, values):
    """{{name}} -> value; an unknown name stays visible as PENDING (never silently empty)"""
    unknown = []

    def sub(m):
        n = m.group(1).strip()
        if n in values:
            return values[n]
        unknown.append(n)
        return PENDING
    return re.sub(r"\{\{([^}]+)\}\}", sub, text), unknown


def read_part(folder, spec, name):
    p = locate(folder, spec, name, "parts")
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def sources_text(folder, spec):
    files = []
    for g in spec.get("sources", []):
        files += sorted(glob.glob(locate(folder, spec, g), recursive=True))
    txt = []
    for f in files:
        try:
            txt.append(open(f, encoding="utf-8", errors="ignore").read())
        except OSError:
            pass
    return "\n".join(txt), files


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def script_summary(path):
    """first line of the module docstring or of the leading comment of a helper script"""
    try:
        t = open(path, encoding="utf-8", errors="ignore").read(4000)
    except OSError:
        return ""
    m = re.search(r'^\s*(?:#!.*\n)?\s*(?:"""|\'\'\')\s*(.+)', t)
    if m:
        return m.group(1).strip().rstrip('"\'').strip()
    m = re.search(r"^\s*(?:#!.*\n)?\s*#\s*(.+)", t)
    return m.group(1).strip() if m else ""


# ------------------------------------------------------------------ render
class State:
    """running numbers of figures and tables"""
    def __init__(self):
        self.fig, self.tab = 0, 0


def figure(inner, caption, st):
    st.fig += 1
    cap = "<figcaption><b>Figure %d.</b> %s</figcaption>" % (st.fig, caption) if caption else ""
    return "<figure>%s%s</figure>" % (inner, cap)


def render_md(md, text, find, out_dir, st, demote=1):
    """find(path) -> the file a picture or video of the text names (locate)"""
    if demote:
        text = re.sub(r"\A\s*#+ [^\n]*\n", "", text)                      # the section title comes from report.json
    text = re.sub(r"^(#+) ", lambda m: "#" * min(6, len(m.group(1)) + demote) + " ", text, flags=re.M)
    html = md.render(text)

    def media(m):                                                         # images are embedded, videos / gifs are copied next to the html
        src, alt = m.group(1), m.group(2)
        f = find(src)
        ext = os.path.splitext(src)[1].lower()
        if not os.path.exists(f):
            return figure('<span class="pend">PENDING</span> (missing: %s)' % esc(src), alt, st)
        if ext in (".mp4", ".webm", ".gif"):
            os.makedirs(os.path.join(out_dir, "media"), exist_ok=True)
            dst = os.path.join(out_dir, "media", os.path.basename(f))
            if os.path.abspath(dst) != os.path.abspath(f):
                shutil.copy(f, dst)
            link = "media/" + os.path.basename(f)
            if ext == ".gif":
                return figure('<img src="%s" alt="%s">' % (link, alt), alt, st)
            return figure('<video controls preload="metadata" src="%s"></video>' % link, alt, st)
        raw = open(f, "rb").read()
        try:
            from PIL import Image
            buf = io.BytesIO()
            Image.open(io.BytesIO(raw)).convert("RGB").save(buf, "JPEG", quality=70, optimize=True, progressive=True)
            raw, mime = buf.getvalue(), "jpeg"
        except Exception:
            mime = "png" if ext == ".png" else "jpeg"
        return figure('<img src="data:image/%s;base64,%s" alt="%s">' % (mime, base64.b64encode(raw).decode(), alt), alt, st)
    html = re.sub(r'(?:<p>)?<img src="([^"]+\.(?:jpg|jpeg|png|gif|mp4|webm))" alt="([^"]*)"[^>]*>(?:</p>)?', media, html)

    def table(m):                                                         # "Table: caption" on the line before a table becomes its numbered caption
        st.tab += 1
        return '<div class="tw"><table><caption><b>Table %d.</b> %s</caption>' % (st.tab, m.group(1))
    html = re.sub(r"<p>Table: (.*?)</p>\s*<table>", table, html, flags=re.S)
    html = re.sub(r"<table>", '<div class="tw"><table>', html)
    html = html.replace("</table>", "</table></div>")
    return html.replace(PENDING, '<span class="pend">PENDING</span>')


def drawings(folder, spec):
    """the architecture drawings of a report: the `svg` parts of its sections (path relative to parts/) that exist, are not empty and are not the scaffold's placeholder.
    Every report holds at least one (the owner, 2026-11-25: "always add an architecture drawing into the reports"); `build --strict` refuses without it"""
    out = []
    for s in spec.get("sections", []):
        svg = read_part(folder, spec, s["svg"]) if s.get("svg") else ""
        if svg.strip() and PLACEHOLDER_SVG not in svg:
            out.append(s["svg"])
    return out


def build(folder, out=None, strict=False):
    from markdown_it import MarkdownIt
    spec = load_spec(folder)
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    values, missing = resolve_values(folder, spec)
    out = out or os.path.join(folder, spec.get("output", "report.html"))
    out_dir = os.path.dirname(os.path.abspath(out))
    st, unknown_all, secs = State(), [], []
    find = functools.partial(locate, folder, spec)

    head = ""                                                             # abstract + teaser (a paper starts with them)
    if spec.get("abstract"):
        text, unk = fill(spec["abstract"], values)
        unknown_all += unk
        head += '<div class="abs"><b class="lab">Abstract</b>%s</div>' % render_md(md, text, find, out_dir, st, demote=0)
    if spec.get("teaser"):
        t = spec["teaser"]
        cap, unk = fill(t.get("caption", ""), values)
        unknown_all += unk
        if str(t.get("image", "")).endswith(".svg"):
            svg = open(find(t["image"]), encoding="utf-8").read() if os.path.exists(find(t["image"])) else ""
            head += figure(re.sub(r"^<\?xml[^>]*\?>\s*", "", svg) or '<span class="pend">PENDING</span>', cap, st)
        else:
            head += render_md(md, "![%s](%s)" % (cap.replace("]", ")"), t.get("image", "")), find, out_dir, st, demote=0)

    for s in spec["sections"]:
        text, unk = fill(read_part(folder, spec, s["part"]), values)
        unknown_all += unk
        fig = ""
        if s.get("svg"):
            svg = read_part(folder, spec, s["svg"])
            cap, unk = fill(s.get("svg_caption", ""), values)
            fig = figure(re.sub(r"^<\?xml[^>]*\?>\s*", "", svg) or '<span class="pend">PENDING</span>', cap, st)
        secs.append((s["id"], s["title"], fig + render_md(md, text, find, out_dir, st, demote=0 if s.get("keep_heading") else 1)))

    util = spec.get("utilities", [])                                      # the helper scripts that are kept
    if util:
        rows = ["Table: The helper scripts that belong to this work and are kept with it (they are in the package under `utilities/`).", "",
                "| script | what it does | how to run it |", "|---|---|---|"]
        for u in util:
            what = u.get("what") or script_summary(find(u["path"])) or PENDING
            rows.append("| `%s` | %s | `%s` |" % (os.path.basename(u["path"]), what, u.get("usage", PENDING)))
        secs.append(("utilities", "B. Utilities", render_md(md, "\n".join(rows), find, out_dir, st, demote=0)))
    if spec.get("sources"):
        _, files = sources_text(folder, spec)
        body = "The numbers of this report come from %d stored source file(s) matched by:\n\n%s\n" % (len(files), "\n".join("- `%s`" % g for g in spec["sources"]))
        secs.append(("sources", "C. Sources of the numbers", render_md(md, body, find, out_dir, st, demote=0)))

    nav = " ".join('<a href="#%s">%s</a>' % (k, t.split(":")[0] if t[0].isdigit() or t[1:2] == "." else t) for k, t, _ in secs)
    body = "\n".join('<section id="%s"><h2>%s</h2>%s</section>' % (k, t, h) for k, t, h in secs if h.strip())
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>%s</title><style>%s</style></head><body><main><h1>%s</h1><div class="sub">%s</div>%s<nav>%s</nav>%s</main></body></html>'
            ) % (esc(spec["title"]), CSS, esc(spec["title"]), esc(spec.get("subtitle", "")), head, nav, body)
    open(out, "w", encoding="utf-8").write(html)
    n_pend = html.count('class="pend"')
    print("%s  %.2f MB  %d figure(s), %d table(s) numbered, %d PENDING mark(s)  (%d value(s) unresolved, %d unknown placeholder(s))"
          % (out, os.path.getsize(out) / 1e6, st.fig, st.tab, n_pend, len(missing), len(unknown_all)))
    for n in missing:
        print("  value not resolved (file or key missing):", n)
    for n in sorted(set(unknown_all)):
        print("  placeholder without a value:", n)
    drawn = drawings(folder, spec)
    if not drawn:
        print("  no architecture drawing: give a section an `svg` part (a file in parts/, the path relative to parts/) that shows the current system, not the scaffold's placeholder")
    if strict and (n_pend or missing or unknown_all or not drawn):
        sys.exit(1)
    return out


# ------------------------------------------------------------------ check
def check(folder, strict=False):
    """every decimal number of the text (abstract, teaser caption, parts; after filling) must occur in the sources: exact, or a rounded form of a source number"""
    spec = load_spec(folder)
    values, _ = resolve_values(folder, spec)
    src, files = sources_text(folder, spec)
    if not files:
        print("no source files matched (report.json 'sources'): nothing to check against")
        return 0
    exact = set(NUM.findall(src))
    srcnums = [float(x) for x in exact]
    texts = [("abstract", spec.get("abstract", "")), ("teaser", spec.get("teaser", {}).get("caption", ""))]
    texts += [(s["part"], read_part(folder, spec, s["part"])) for s in spec["sections"]]
    bad, total = [], 0
    for name, raw in texts:
        text, _ = fill(raw, values)
        text = re.sub(r"```.*?```", "", text, flags=re.S)                     # code blocks hold commands, not results
        for ln, line in enumerate(text.split("\n"), 1):
            if line.lstrip().startswith("#"):                                   # headings carry section numbers, not results
                continue
            for m in NUM.finditer(line):
                tok = m.group(0)
                total += 1
                if tok in exact:
                    continue
                d = len(tok.split(".")[1])
                v = float(tok)
                if any(abs(round(x, d) - v) < 1e-12 for x in srcnums):
                    continue
                bad.append((name, ln, tok, line.strip()[:110]))
    print("%d decimal numbers in the text, %d not found in %d source file(s)" % (total, len(bad), len(files)))
    for p, ln, tok, line in bad[:60]:
        print("  %s:%d  %s   | %s" % (p, ln, tok, line))
    if len(bad) > 60:
        print("  ... %d more" % (len(bad) - 60))
    if strict and bad:
        sys.exit(1)
    return len(bad)


# ------------------------------------------------------------------ pack
def pack(folder, out=None):
    spec = load_spec(folder)
    pk = spec.get("pack", {})
    limit = int(pk.get("limit", 2_000_000))
    html = build(folder, os.path.join(folder, spec.get("output", "report.html")))
    out = out or os.path.join(folder, spec.get("package", "package.zip"))
    code = []
    for g in pk.get("code", []):
        code += [f for f in sorted(glob.glob(locate(folder, spec, g), recursive=True)) if os.path.isfile(f) and "__pycache__" not in f and not f.endswith(".pyc")]
    strip = os.path.abspath(locate(folder, spec, pk["strip"])) if pk.get("strip") else None
    code = [os.path.abspath(f) for f in code]
    utils = [f for f in (locate(folder, spec, u["path"]) for u in spec.get("utilities", [])) if os.path.isfile(f)]
    if pk.get("require_clean_git") and (code or utils):
        for f in code + utils:
            dirty = subprocess.run(["git", "status", "--porcelain", "--", f], capture_output=True, text=True, cwd=os.path.dirname(f)).stdout.strip()
            if dirty:
                sys.exit("refused: uncommitted change in %s" % f)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        if pk.get("include_report", True):               # False: the zip holds code + utilities only (the html is sent next to it)
            z.write(html, "report.html")
        media = os.path.join(os.path.dirname(os.path.abspath(html)), "media")
        if os.path.isdir(media):
            for f in sorted(os.listdir(media)):
                z.write(os.path.join(media, f), "media/" + f)
        if pk.get("readme"):
            z.writestr("README.txt", pk["readme"])
        for f in code:
            name = os.path.relpath(f, strip) if strip and f.startswith(strip + os.sep) else os.path.basename(f)
            z.write(f, "code/" + name.replace("\\", "/"))
        for f in utils:
            z.write(f, "utilities/" + os.path.basename(f))
    n = os.path.getsize(out)
    if not pk.get("include_report", True):
        n += os.path.getsize(html)                       # the limit then counts report + package together
    print("%s  %d bytes (%.2f MB), %d code file(s), %d utilitie(s)" % (out, n, n / 1e6, len(code), len(utils)))
    if n >= limit:
        sys.exit("refused: package is %d bytes, limit %d" % (n, limit))
    return out


# ------------------------------------------------------------------ new
def new(folder, title):
    os.makedirs(os.path.join(folder, "parts"), exist_ok=True)
    os.makedirs(os.path.join(folder, "media"), exist_ok=True)
    spec = {
        "title": title, "subtitle": "Project, date, status: PENDING",
        "abstract": "PENDING: four sentences: the problem, what was built, the headline result against its baseline ({{headline}}), what is still open.",
        "teaser": {"image": "parts/teaser.jpg", "caption": "PENDING: one real example, input and output side by side; say what to look at."},
        "sections": [{"id": i, "title": t, "part": p, **({"svg": "02_system.svg", "svg_caption": "PENDING: what the drawing shows and how to read it."} if i == "system" else {})}
                     for i, t, p, _ in CHAPTERS],
        "values": {"headline": {"file": "../path/to/metrics.json", "key": "topo_f1", "fmt": "{:.4f}"}},
        "sources": ["../path/to/cards/**/*.md", "../path/to/results/**/*.json"],
        "utilities": [{"path": "../path/to/tools/helper.py", "what": "", "usage": "python helper.py --help"}],
        "output": "report.html", "package": "package.zip",
        "pack": {"limit": 2000000, "code": ["../path/to/code/**/*.py"], "strip": "../path/to/code", "require_clean_git": False,
                 "readme": "What is in this package, in three lines. The weights are not in it (size limit).\n"},
    }
    json.dump(spec, open(os.path.join(folder, "report.json"), "w", encoding="utf-8"), indent=1)
    for i, t, p, prompt in CHAPTERS:
        path = os.path.join(folder, "parts", p)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").write("# %s\n\n%s\n" % (t, prompt))
    svg = os.path.join(folder, "parts", "02_system.svg")
    if not os.path.exists(svg):
        open(svg, "w", encoding="utf-8").write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 60"><text x="10" y="35" font-size="14">%s</text></svg>\n' % PLACEHOLDER_SVG)
    print("scaffold in", folder, "- edit report.json (abstract, teaser, values, sources, utilities, pack) and the parts, then: build, check, pack")


# ------------------------------------------------------------------ selftest
def selftest():
    d = tempfile.mkdtemp(prefix="reporter_selftest_")
    try:
        rep = os.path.join(d, "rep")
        new(rep, "Selftest report")
        os.makedirs(os.path.join(d, "res")), os.makedirs(os.path.join(d, "code")), os.makedirs(os.path.join(d, "tools"))
        json.dump({"a": {"topo": [0.5486, 0.5743]}}, open(os.path.join(d, "res", "m.json"), "w"))
        open(os.path.join(d, "res", "card.md"), "w").write("result 0.5486 and 0.5743 and 0.6912 and 0.123456\n")
        open(os.path.join(d, "code", "x.py"), "w").write("print(1)\n")
        open(os.path.join(d, "tools", "helper.py"), "w").write('"""Helper: prints one.\nmore"""\nprint(1)\n')
        spec = load_spec(rep)
        spec["values"] = {"s0": {"file": "../res/m.json", "key": "a.topo.0", "fmt": "{:.4f}"}, "s1": {"file": "../res/m.json", "key": "a.topo.1", "fmt": "{:.4f}"},
                          "gone": {"file": "../res/nope.json", "key": "x"}}
        spec["abstract"] = "Topo F1 {{s0}} / {{s1}}."
        spec["teaser"] = {"image": "media/shot.png", "caption": "A real frame."}
        spec["sources"] = ["../res/*.md", "../res/*.json"]
        spec["utilities"] = [{"path": "../tools/helper.py", "usage": "python helper.py"}]
        spec["pack"]["code"], spec["pack"]["strip"] = ["../code/*.py"], "../code"
        json.dump(spec, open(os.path.join(rep, "report.json"), "w"), indent=1)
        open(os.path.join(rep, "parts", "05_quant.md"), "w").write(
            "# Q\n\nTable: Topo F1 on VAL.\n\n| split | topo F1 |\n|---|---|\n| VAL s0 / s1 | {{s0}} / {{s1}} |\n| missing | {{gone}} |\n\n"
            "rounded 0.549, exact 0.6912, wrong 0.7777, unknown {{nope}}.\n\n![A clip of a drive.](media/clip.mp4)\n")
        open(os.path.join(rep, "media", "clip.mp4"), "wb").write(b"\0" * 100)
        record = {"numbers": {"graph.topo_F1": 0.5, "graph": {"n": 1}}, "seeds": [{"seg.iou@0.5": 2}]}
        assert dig(record, "numbers.graph.topo_F1") == 0.5 and dig(record, "numbers.graph.n") == 1 and dig(record, "seeds.0.seg.iou@0.5") == 2, "keys with dots"
        out = build(rep)
        html = open(out, encoding="utf-8").read()
        assert "Topo F1 0.5486 / 0.5743." in html, "abstract / values not filled"
        assert "<b>Table 1.</b> Topo F1 on VAL." in html, "table caption"
        assert "<b>Figure 1.</b> A real frame." in html and "<b>Figure 3.</b> A clip of a drive." in html and "<video" in html, "figure numbering / video"
        assert html.count('class="pend"') >= 3, "PENDING marks missing"
        assert "helper.py" in html and "Helper: prints one." in html, "utilities table (docstring summary)"
        assert os.path.exists(os.path.join(rep, "media", "clip.mp4")), "video media"
        bad = check(rep)
        assert bad >= 1, "the wrong number 0.7777 must be found by check"
        z = pack(rep)
        names = zipfile.ZipFile(z).namelist()
        assert {"report.html", "code/x.py", "README.txt", "utilities/helper.py"} <= set(names), names
        try:
            build(rep, strict=True)
            raise AssertionError("strict build must fail with PENDING marks")
        except SystemExit:
            pass
        assert drawings(rep, spec) == [], "the scaffold's placeholder svg is no drawing"
        open(os.path.join(rep, "parts", "02_system.svg"), "w").write('<svg xmlns="http://www.w3.org/2000/svg"><rect width="9" height="9"/></svg>\n')
        assert drawings(rep, spec) == ["02_system.svg"], "a real svg part is the drawing"
        flat = dict(spec, sections=[s for s in spec["sections"] if not s.get("svg")])
        assert drawings(rep, flat) == [], "a report without an svg part has no drawing"
        # the strict gate itself: a clean report (no PENDING, all values) fails without a drawing and passes with one
        clean = os.path.join(d, "clean")
        os.makedirs(os.path.join(clean, "parts"))
        sec = {"id": "system", "title": "2. The system", "part": "02_system.md"}            # no `svg` part: nothing else is PENDING, so only the gate can refuse
        json.dump({"title": "Clean", "sections": [sec]}, open(os.path.join(clean, "report.json"), "w"))
        open(os.path.join(clean, "parts", "02_system.md"), "w").write("The chain.\n")
        try:
            build(clean, strict=True)
            raise AssertionError("strict build must fail without a drawing")
        except SystemExit:
            pass
        json.dump({"title": "Clean", "sections": [dict(sec, svg="02_system.svg")]}, open(os.path.join(clean, "report.json"), "w"))
        open(os.path.join(clean, "parts", "02_system.svg"), "w").write('<svg xmlns="http://www.w3.org/2000/svg"><rect width="9" height="9"/></svg>\n')
        build(clean, strict=True)
        # a copy of a report folder (research report): what lies outside the folder is read from `origin`, whatever names it
        src, copy = os.path.join(d, "deep", "src"), os.path.join(d, "a", "b", "c", "copy")
        os.makedirs(os.path.join(src, "parts"))
        open(os.path.join(d, "res", "draw.svg"), "w").write('<svg xmlns="http://www.w3.org/2000/svg"><rect width="7" height="7"/></svg>\n')
        open(os.path.join(d, "res", "pic.png"), "wb").write(b"\x89PNG not really")
        spec = {"title": "Copy", "abstract": "Topo F1 {{v}}.", "teaser": {"image": "../../res/pic.png", "caption": "Teaser."},
                "sections": [{"id": "s", "title": "2. The system", "part": "02_system.md", "svg": "../../../res/draw.svg"}],
                "values": {"v": {"file": "../../res/m.json", "key": "a.topo.0"}}, "sources": ["../../res/*.md"],
                "utilities": [{"path": "../../tools/helper.py", "usage": "python helper.py"}], "pack": {"code": ["../../code/*.py"], "strip": "../../code"}}
        open(os.path.join(src, "parts", "02_system.md"), "w").write("Exact 0.6912.\n\n![A picture.](../../res/pic.png)\n")
        json.dump(spec, open(os.path.join(src, "report.json"), "w"))
        shutil.copytree(src, copy)
        try:
            build(copy, strict=True)
            raise AssertionError("a copy without origin must not find what lies outside it")
        except SystemExit:
            pass
        json.dump(dict(spec, origin=src), open(os.path.join(copy, "report.json"), "w"))
        html = open(build(copy, strict=True), encoding="utf-8").read()
        assert html.count("data:image/png") == 2 and 'width="7"' in html and "Topo F1 0.5486." in html and "Helper: prints one." in html, "outside paths of a copy"
        assert check(copy) == 0, "the sources of a copy"
        assert {"code/x.py", "utilities/helper.py"} <= set(zipfile.ZipFile(pack(copy)).namelist()), "the package of a copy"
        print("selftest OK")
    finally:
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("new"); a.add_argument("folder"); a.add_argument("--title", default="Report")
    for c in ("build", "check", "pack"):
        b = sp.add_parser(c); b.add_argument("folder"); b.add_argument("--out"); b.add_argument("--strict", action="store_true")
    sp.add_parser("selftest")
    x = ap.parse_args()
    if x.cmd == "new":
        new(x.folder, x.title)
    elif x.cmd == "build":
        build(x.folder, x.out, x.strict)
    elif x.cmd == "check":
        check(x.folder, x.strict)
    elif x.cmd == "pack":
        pack(x.folder, x.out)
    else:
        selftest()
