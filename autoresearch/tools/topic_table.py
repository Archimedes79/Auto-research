"""The table of a research topic, made from its notes (never typed by an agent).

    python autoresearch/tools/topic_table.py <topic folder> [--min-relevance N]    # writes table.md and table.csv, exit 1 on a bad note
    python autoresearch/tools/topic_table.py --selftest

A topic folder holds one subfolder per paper (with its code), each with a `note.md` written by one agent. Right after the `# ` line of a note come exactly
these seven lines, in this order, one per line, nothing between them (blank lines and one-line <!-- comments --> before the first are fine):

    - title: the title of the paper
    - relevance: 0-3 (3 = take it into the plan now)
    - summary: about three sentences: what it does and why it matters for us
    - status: checked (page read) / digest / memory
    - paper: https://... (a plain link: no spaces, quotes, brackets, parentheses, pipes or backslashes; percent-encode them)
    - code: https://... or -
    - verdict: use / watch / context / not relevant

The table has one row per note, best relevance first: title (a link to the note), summary, relevance, status, paper, code. A note that breaks a rule,
a subfolder without `note.md`, an empty topic folder or two notes on the same paper make the tool print the reasons, REMOVE any older table.md /
table.csv (so a stale table never looks current) and exit 1. With --min-relevance N the files are table_min<N>.md / .csv and table.md stays the table of all notes.
"""
import argparse
import csv
import io
import os
import re
import sys

KEYS = ("title", "relevance", "summary", "status", "paper", "code", "verdict")
VERDICTS = ("use", "watch", "context", "not relevant")
STATUS = ("checked", "digest", "memory")
MAX_SUMMARY = 700  # characters; about three sentences
URL = re.compile(r"https?://[^\s<>\"'`()|\[\]\\]+$")
FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")


def header(text):
    """({key: value}, [problems]) of the header block right after the '# ' line"""
    lines = text.lstrip("﻿").replace("\r\n", "\n").split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not lines[i].startswith("# "):
        return {}, ["the first line is not a '# ' title line"]
    i += 1
    while i < len(lines) and (not lines[i].strip() or re.fullmatch(r"<!--.*-->", lines[i].strip())):
        i += 1
    got, order = {}, []
    while i < len(lines):
        m = re.match(r"^- (\w+):[ \t]*(.*?)[ \t]*$", lines[i])
        if not m:
            break
        order.append(m.group(1))
        got.setdefault(m.group(1), m.group(2))
        i += 1
    bad = []
    if order != list(KEYS):
        bad.append("the header must be exactly these lines in this order right after the title: " + ", ".join(KEYS) + " (found: " + (", ".join(order) or "none") + ")")
    return got, bad


def norm_url(u):
    """one spelling of a paper link for the duplicate check: scheme and www dropped, no fragment, no trailing slash, arXiv abs / pdf / version one id"""
    u = re.sub(r"^https?://(www\.)?", "", u.lower()).split("#")[0].rstrip("/")
    m = re.match(r"arxiv\.org/(?:abs|pdf)/([^?]+?)(?:v\d+)?(?:\.pdf)?$", u)
    return "arxiv:" + m.group(1) if m else u


def problems(h):
    """what is wrong with the values of one header (a list of words)"""
    bad = []
    if not re.fullmatch(r"[0-3]", h.get("relevance", "")):
        bad.append("relevance is not 0-3")
    for k in ("title", "summary"):
        if not h.get(k):
            bad.append(f"{k} is empty")
    if not URL.match(h.get("paper", "")):
        bad.append("paper is not a plain http(s) link")
    if h.get("code") != "-" and not URL.match(h.get("code", "")):
        bad.append("code is not a plain http(s) link or -")
    if h.get("verdict") not in VERDICTS:
        bad.append("verdict is not one of " + " / ".join(VERDICTS))
    if h.get("status") not in STATUS:
        bad.append("status is not one of " + " / ".join(STATUS))
    if len(h.get("summary", "")) > MAX_SUMMARY:
        bad.append(f"summary is longer than {MAX_SUMMARY} characters")
    return bad


def collect(folder):
    """(rows, errors): a row per note, errors as [(where, [words])]"""
    rows, errors = [], []
    root = os.path.realpath(folder)
    for name in sorted(os.listdir(folder)):
        sub = os.path.join(folder, name)
        if not os.path.isdir(sub) or name.startswith((".", "_")):
            continue
        note = os.path.join(sub, "note.md")
        if not FOLDER.match(name):
            errors.append((name, ["the folder name may hold only letters, digits, . _ -"]))
        elif os.path.islink(sub) or not os.path.realpath(note).startswith(root + os.sep):
            errors.append((name, ["a link: the note must lie inside the topic folder"]))
        elif not os.path.isfile(note):
            errors.append((name, ["no note.md in this folder"]))
        else:
            try:
                h, bad = header(open(note, encoding="utf-8").read())
            except (OSError, ValueError) as e:
                errors.append((name, [f"unreadable: {e}"]))
                continue
            bad += [] if bad else problems(h)
            if bad:
                errors.append((name, bad))
            else:
                rows.append(dict(h, id=name, link=f"{name}/note.md"))
    seen = {}
    for r in rows:
        k = norm_url(r["paper"])
        if k in seen:
            errors.append((r["id"], [f"same paper as {seen[k]} (a code repository belongs into the note of its paper)"]))
        seen.setdefault(k, r["id"])
    if not rows and not errors:
        errors.append((folder, ["no notes (no subfolder with a note.md)"]))
    return rows, errors


def md_text(s):
    """text of a table cell: no pipe, no markup that could make a link or tag"""
    s = s.replace("\\", "\\\\").replace("|", "\\|").replace("[", "\\[").replace("]", "\\]").replace("`", "\\`").replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("\n", " ")


def csv_cell(s):
    """a cell that a spreadsheet would run as a formula gets a leading quote (the plain code value '-' stays)"""
    return "'" + s if s != "-" and s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def render(rows, min_relevance=0):
    rows = [r for r in rows if int(r["relevance"]) >= min_relevance]
    rows.sort(key=lambda r: (-int(r["relevance"]), r["title"].lower(), r["id"]))
    md = ["# Table of the topic" + (f" (relevance >= {min_relevance})" if min_relevance else ""), "",
          "Made by `autoresearch/tools/topic_table.py` from the notes; do not edit.", "",
          "| title | summary | relevance | status | paper | code |", "|---|---|---|---|---|---|"]
    for r in rows:
        code = f"[code]({r['code']})" if r["code"] != "-" else "-"
        md.append(f"| [{md_text(r['title'])}]({r['link']}) | {md_text(r['summary'])} | {r['relevance']} | {r['status']} | [paper]({r['paper']}) | {code} |")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "title", "summary", "relevance", "status", "verdict", "paper", "code", "note"])
    for r in rows:
        w.writerow([r["id"], csv_cell(r["title"]), csv_cell(r["summary"]), r["relevance"], r["status"], r["verdict"], r["paper"], r["code"], r["link"]])
    return "\n".join(md) + "\n", buf.getvalue()


def run(folder, min_relevance=0):
    """write the table; (exit code, messages)"""
    rows, errors = collect(folder)
    stem = "table" if not min_relevance else f"table_min{min_relevance}"
    if errors:
        gone = []
        for ext in ("md", "csv"):  # the generated tables of an earlier state are removed, so none looks current
            p = os.path.join(folder, f"table.{ext}")
            if os.path.isfile(p):
                os.remove(p)
                gone.append(os.path.basename(p))
        msgs = [f"{w}: {'; '.join(why)}" for w, why in errors]
        msgs.append(f"{len(errors)} problem(s): nothing written" + (f"; removed the old {', '.join(gone)}" if gone else ""))
        return 1, msgs
    md, csv_text = render(rows, min_relevance)
    open(os.path.join(folder, stem + ".md"), "w", encoding="utf-8", newline="\n").write(md)
    open(os.path.join(folder, stem + ".csv"), "w", encoding="utf-8", newline="").write(csv_text)
    return 0, [f"{len(rows)} notes -> {os.path.join(folder, stem + '.md')} and .csv"]


def selftest():
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="ar_topic_")
    try:
        def note(name, body):
            os.makedirs(os.path.join(d, name), exist_ok=True)
            open(os.path.join(d, name, "note.md"), "w", encoding="utf-8", newline="").write(body)

        ok = ("# {n} t\n\n<!-- a comment line is fine -->\n- title: {t}\n- relevance: {r}\n- summary: It does a. It matters because b.\n- status: checked\n"
              "- paper: https://x.org/{n}\n- code: {c}\n- verdict: use\n\n## More\n- code: https://body.example/ignored\n")
        note("zz1", ok.format(n="zz1", t="Alpha | [pipe] <b>", r="1", c="-"))
        note("yy2", ok.format(n="yy2", t="Beta", r="3", c="https://github.com/b/b"))
        note("xx3", ok.format(n="xx3", t="Aardvark", r="3", c="-"))
        rows, errors = collect(d)
        assert not errors and sorted(r["id"] for r in rows) == ["xx3", "yy2", "zz1"], (rows, errors)
        md, csv_text = render(rows)
        lines = [ln for ln in md.splitlines() if ln.startswith("| [")]
        assert [ln.split("]")[0] for ln in lines][:2] == ["| [Aardvark", "| [Beta"], lines  # relevance first, then title (ids run the other way: the title decides)
        assert "Alpha \\| \\[pipe\\] &lt;b&gt;" in lines[2] and "[code](https://github.com/b/b)" in md and "body.example" not in md, lines[2]
        assert csv_text.splitlines()[0].startswith("id,title") and len(csv_text.splitlines()) == 4
        assert len([ln for ln in render(rows, 2)[0].splitlines() if ln.startswith("| [")]) == 2, "min relevance"
        # CRLF and a BOM are fine
        note("bom1", "﻿" + ok.format(n="bom1", t="Bom", r="2", c="-").replace("\n", "\r\n"))
        assert "bom1" in [r["id"] for r in collect(d)[0]], "BOM and CRLF"
        shutil.rmtree(os.path.join(d, "bom1"))
        # every kind of bad note is named with its reason
        base = ok.format(n="q", t="Q", r="1", c="-")
        cases = {
            "badval": ("# d\n- title: D\n- relevance: 5\n- summary: s\n- status: guess\n- paper: ftp://x\n- code: nowhere\n- verdict: maybe\n", ["relevance is not 0-3", "paper is not", "code is not", "verdict is not", "status is not"]),
            "short": ("# e\n- title: Eps\n", ["header must be exactly"]),
            "order": ("# o\n- relevance: 1\n- title: T\n- summary: s\n- status: checked\n- paper: https://x.org/o\n- code: -\n- verdict: use\n", ["header must be exactly"]),
            "dupkey": (base.replace("- code: -\n", "- code: -\n- title: again\n", 1), ["header must be exactly"]),
            "indent": (base.replace("- summary:", "  - summary:", 1), ["header must be exactly"]),
            "aftertable": ("# t\n\n| a | b |\n|---|---|\n\n" + base.split("\n", 1)[1], ["header must be exactly"]),
            "pipeurl": (base.replace("https://x.org/q", "https://x.org/a|b"), ["paper is not a plain"]),
            "parenurl": (base.replace("https://x.org/q", "https://x.org/w_(y)"), ["paper is not a plain"]),
            "scheme": (base.replace("https://x.org/q", "javascript:alert(1)"), ["paper is not a plain"]),
            "long": (base.replace("It does a. It matters because b.", "x" * 800), ["longer than"]),
            "notitle": ("- title: T\n", ["not a '# ' title line"]),
            "dup": (ok.format(n="zz1", t="Same paper as zz1", r="2", c="-").replace("x.org/zz1", "x.org/zz1/#sec"), []),
        }
        for name, (body, want) in cases.items():
            note(name, body)
        os.makedirs(os.path.join(d, "nonote"))
        open(os.path.join(d, "nonote", "readme.md"), "w").write("x")
        note("bad name", ok.format(n="s", t="S", r="1", c="-"))
        os.makedirs(os.path.join(d, "_work"))  # a folder with a leading underscore is not a paper
        rows, errors = collect(d)
        got = dict(errors)
        for name, (body, want) in cases.items():
            if name != "dup":
                assert name in got and all(any(w in g for g in got[name]) for w in want), (name, got.get(name))
        assert any("same paper as dup" in g for g in got["zz1"]), "the later of two notes on one paper (the # fragment and a trailing slash do not matter) is named, got: " + str(got)
        assert "no note.md" in got["nonote"][0] and "folder name" in got["bad name"][0] and "_work" not in got, got
        # the arXiv spellings of one paper are one paper
        assert norm_url("https://arxiv.org/abs/2401.01234v2") == norm_url("http://www.arxiv.org/pdf/2401.01234.pdf") == norm_url("https://arxiv.org/abs/2401.01234/")
        # the command: a bad folder removes an older table and exits 1, a clean one writes both files
        open(os.path.join(d, "table.md"), "w").write("old")
        open(os.path.join(d, "table.csv"), "w").write("old")
        code, msgs = run(d)
        assert code == 1 and not os.path.exists(os.path.join(d, "table.md")) and not os.path.exists(os.path.join(d, "table.csv")) and "removed the old table.md, table.csv" in msgs[-1], msgs
        for name in list(cases) + ["nonote", "bad name", "_work"]:
            shutil.rmtree(os.path.join(d, name))
        code, msgs = run(d)
        assert code == 0 and os.path.isfile(os.path.join(d, "table.md")) and os.path.isfile(os.path.join(d, "table.csv")), msgs
        code, msgs = run(d, 2)
        assert code == 0 and os.path.isfile(os.path.join(d, "table_min2.md")) and "Aardvark" in open(os.path.join(d, "table.md")).read() and "Alpha" in open(os.path.join(d, "table.md")).read() \
            and "Alpha" not in open(os.path.join(d, "table_min2.md")).read(), "a filtered table has its own file and table.md stays complete"
        # a spreadsheet formula is made harmless, the plain code value stays
        assert csv_cell("=HYPERLINK(1)") == "'=HYPERLINK(1)" and csv_cell("-") == "-" and csv_cell("@x") == "'@x" and csv_cell("plain") == "plain"
        # an empty folder and a note that is not UTF-8
        empty = tempfile.mkdtemp(prefix="ar_topic_empty_")
        try:
            code, msgs = run(empty)
            assert code == 1 and "no notes" in msgs[0], msgs
            os.makedirs(os.path.join(empty, "b1"))
            open(os.path.join(empty, "b1", "note.md"), "wb").write(b"# b\n\xff\xfe")
            code, msgs = run(empty)
            assert code == 1 and "unreadable" in msgs[0], msgs
        finally:
            shutil.rmtree(empty)
        # a link out of the folder is refused
        out = tempfile.mkdtemp(prefix="ar_topic_out_")
        try:
            open(os.path.join(out, "note.md"), "w").write(ok.format(n="o", t="O", r="1", c="-"))
            if hasattr(os, "symlink"):
                os.symlink(out, os.path.join(d, "linked"))
                assert "link" in dict(collect(d)[1])["linked"][0]
                os.remove(os.path.join(d, "linked"))
        finally:
            shutil.rmtree(out)
    finally:
        shutil.rmtree(d)
    print("topic_table selftest passed")


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--min-relevance", type=int, default=0, choices=[0, 1, 2, 3])
    a = ap.parse_args(argv)
    if not os.path.isdir(a.folder):
        sys.exit(f"{a.folder} is not a folder")
    code, msgs = run(a.folder, a.min_relevance)
    print("\n".join(msgs))
    sys.exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
