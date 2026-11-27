"""The table of a research topic, made from its notes (never typed by an agent).

    python3 literature/topic_table.py <topic> [--min-relevance N] [--known FILE ...]    # writes table.md and table.csv, exit 1 on a bad note
    python3 literature/topic_table.py <topic> --set-result <note id> "E398: CUT topo F1 -0.005" [--all]   # fill the result line of a note, then the table
    python3 literature/topic_table.py <topic> --check [--all-topics]    # the done state of the procedure, writes nothing, exit 1 on any problem
    python3 literature/topic_table.py --selftest

(from autoresearch/; plain Python 3.) <topic> is the topic's folder, or <problem>/<slug> for problems/<problem>/literature/topics/<slug>/. One tool for
every problem: the notes and tables are the problem's own data, the procedure (README.md next to this file) and the tool are shared.
A topic folder holds one subfolder per paper (with its code), each with a `note.md` written by one agent. Right after the `# ` line of a note come exactly
these eight lines, in this order, one per line, nothing between them (blank lines and one-line <!-- comments --> before the first are fine):

    - title: the title of the paper
    - relevance: 0-3 (3 = take it into the plan now)
    - summary: about three sentences: what it does and why it matters for us
    - status: checked (page read) / digest / memory
    - paper: https://... (a plain link: no spaces, quotes, brackets, parentheses, pipes or backslashes; percent-encode them)
    - code: https://... or -
    - verdict: use / watch / context / not relevant
    - result: - (no experiment yet), or "<experiment id>: KEEP / CUT / TIE / INCONCLUSIVE / INVALID / MEASURED / planned / running" and a few words (the number); filled with --set-result when a card that cites the paper is judged

The table has one row per note, best relevance first: title (a link to the note), summary, relevance, status, paper, code, experiment result. A note that breaks a rule,
a subfolder without `note.md`, an empty topic folder or two notes on the same paper make the tool print the reasons, REMOVE the older tables (table.md / .csv and
every table_min<N>.md / .csv; only files this tool wrote, so a stale table never looks current) and exit 1. With --min-relevance N the files are table_min<N>.md / .csv and table.md stays
the table of all notes.
--known <file> (several allowed): what the problem already holds. A papers table (csv: the column that holds the paper link is the one named url / link /
paper, else the one filled with http(s) links; a row without a link whose id is an arXiv id counts by that id; no such column: exit 2) or a markdown file
with tables, such as a catalogue or a technique register (every http(s) link of a row counts, under the row's first cell). The notes whose paper is in one
of them are listed, and the table gets a last column `known` (yes / -). Without it the table has no such column.
--set-result ... --all: the same paper may be a note in several topics of the problem; the result line is written into every note on that paper (same
normalised link) in every topic of the same topics folder and the table of each touched topic is made again. Without --all only the named note changes.
--check: the done state of the procedure, nothing is written, every reason is printed, exit 1 if there is one: the notes are valid (as in a normal run); table.md
and table.csv hold what the notes give now; every relative link of overview.md and table.md resolves, and overview.md links at least one note; check.md has
one bullet line per claim, "- confirmed: <note id>: ...", "- contradicted: ..." or "- unreadable: ...", at least one confirmed claim, no contradicted or
unreadable claim of a paper that is still in the topic (claims about a paper moved to _rejected/ no longer count), and no note is newer than it; every id of
the merged list candidates.md (the candidates_<subtopic>.md files only while there is no merge; only tables whose first column is `id`) has a folder or lies
under _rejected/, except the rows whose `known?` column says yes, and there is at least one candidate; no note sits in a folder that starts with _ or . (the
table tool skips those; _rejected itself is allowed). --all-topics adds a warning (no error) for a paper that is a note in more than one topic of that folder.
"""
import argparse
import contextlib
import csv
import io
import os
import re
import sys
from urllib.parse import unquote

KEYS = ("title", "relevance", "summary", "status", "paper", "code", "verdict", "result")
RESULT = re.compile(r"(-|[A-Za-z0-9][A-Za-z0-9._-]*: (KEEP|CUT|TIE|INCONCLUSIVE|INVALID|MEASURED|planned|running)( [^\n]{1,160})?)$")
VERDICTS = ("use", "watch", "context", "not relevant")
STATUS = ("checked", "digest", "memory")
MAX_SUMMARY = 700  # characters; about three sentences
URL = re.compile(r"https?://[^\s<>\"'`()|\[\]\\]+$")
FOLDER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*$")
TABLES = re.compile(r"table(_min\d+)?\.(md|csv)$")  # the files this tool writes (nothing else is removed)
LINK_NAMES = ("url", "link", "paper", "paper_url", "paper_link")  # names of the link column of a papers table
ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}(v\d+)?$")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # autoresearch/literature/ -> the repository


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
    u = re.sub(r"^https?://(www\.|export\.)?", "", u.lower()).split("#")[0].rstrip("/")
    m = re.match(r"(?:ar5iv\.labs\.)?arxiv\.org/(?:abs|pdf|html)/([^?]+?)(?:v\d+)?(?:\.pdf)?$", u)
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
    if not RESULT.match(h.get("result", "")):
        bad.append("result is not '-' or '<experiment id>: KEEP / CUT / TIE / INCONCLUSIVE / INVALID / MEASURED / planned / running [words]'")
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


def load_known(path):
    """{normalised paper link: id of the row} of what a problem already holds; ValueError if the file cannot be read or holds no paper link.
    A markdown file (catalogue, register): every http(s) link of a table row counts, under the row's first cell. A papers table (csv): the link
    column is the one named like a link (url, link, paper, ...), else the one whose filled cells are (nearly) all http(s) links; a row without a
    link whose id is an arXiv id counts by that id (the lane table keeps old papers that way)."""
    if path.lower().endswith(".md"):
        try:
            with open(path, encoding="utf-8-sig") as f:
                lines = f.read().splitlines()
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read {path}: {e}")
        known = {}
        for ln in lines:
            if ln.startswith("|"):
                rid = ln.strip().strip("|").split("|")[0].strip().strip("*` ")
                for link in re.findall(r"https?://[^\s<>\"'`()|\[\]\\]+", ln):
                    known.setdefault(norm_url(link.rstrip(".,;")), rid or link)
        if not known:
            raise ValueError(f"{path}: no table row holds a link")
        return known
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields, table = [c for c in (reader.fieldnames or []) if c], list(reader)
    except (OSError, ValueError, csv.Error) as e:
        raise ValueError(f"cannot read {path}: {e}")

    def filled(col):
        return [(r.get(col) or "").strip() for r in table if (r.get(col) or "").strip()]

    def links(col):
        return [c for c in filled(col) if re.match(r"https?://", c, re.I)]
    named = [c for c in fields if re.sub(r"[\s-]+", "_", c.strip().lower()) in LINK_NAMES and links(c)]
    content = [c for c in fields if links(c) and len(links(c)) >= 0.9 * len(filled(c))]
    if not (named or content):
        raise ValueError(f"{path}: no column that holds paper links (a column named url / link / paper, or one filled with http(s) links)")
    col, known = (named or content)[0], {}
    for r in table:
        link, rid = (r.get(col) or "").strip(), (r.get("id") or "").strip()
        if not link and ARXIV_ID.match(rid):
            link = "https://arxiv.org/abs/" + rid
        if link:
            known.setdefault(norm_url(link), rid or link)
    return known


def render(rows, min_relevance=0, known=None):
    """(table.md text, table.csv text); with `known` (load_known) a last column says whether the paper is in the problem's papers table"""
    rows = [r for r in rows if int(r["relevance"]) >= min_relevance]
    rows.sort(key=lambda r: (-int(r["relevance"]), r["title"].lower(), r["id"]))
    cols = ["title", "summary", "relevance", "status", "paper", "code", "experiment result"] + (["known"] if known is not None else [])
    md = ["# Table of the topic" + (f" (relevance >= {min_relevance})" if min_relevance else ""), "",
          "Made by `autoresearch/literature/topic_table.py` from the notes; do not edit.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    yes = lambda r: "yes" if norm_url(r["paper"]) in known else "-"  # noqa: E731
    for r in rows:
        code = f"[code]({r['code']})" if r["code"] != "-" else "-"
        md.append(f"| [{md_text(r['title'])}]({r['link']}) | {md_text(r['summary'])} | {r['relevance']} | {r['status']} | [paper]({r['paper']}) | {code} | {md_text(r['result'])} |"
                  + (f" {yes(r)} |" if known is not None else ""))
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "title", "summary", "relevance", "status", "verdict", "paper", "code", "result", "note"] + (["known"] if known is not None else []))
    for r in rows:
        w.writerow([r["id"], csv_cell(r["title"]), csv_cell(r["summary"]), r["relevance"], r["status"], r["verdict"], r["paper"], r["code"], csv_cell(r["result"]), r["link"]]
                   + ([yes(r)] if known is not None else []))
    return "\n".join(md) + "\n", buf.getvalue()


def generated(path):
    """is this file a table the tool wrote (its fixed first lines)? Another file of that name is never removed"""
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(400)
    except (OSError, ValueError):
        return False
    return head.startswith("# Table of the topic") if path.endswith(".md") else head.startswith("id,title,summary,relevance,status")


def run(folder, min_relevance=0, known=None):
    """write the table; (exit code, messages)"""
    rows, errors = collect(folder)
    stem = "table" if not min_relevance else f"table_min{min_relevance}"
    if errors:
        # the generated tables of an earlier state (all of them, also the filtered ones) are removed, so none looks current
        old = sorted((n for n in os.listdir(folder) if TABLES.match(n) and generated(os.path.join(folder, n))), key=lambda n: (n.rsplit(".", 1)[0], n.endswith("csv")))
        for n in old:
            os.remove(os.path.join(folder, n))
        msgs = [f"{w}: {'; '.join(why)}" for w, why in errors]
        msgs.append(f"{len(errors)} problem(s): nothing written" + (f"; removed the old {', '.join(old)}" if old else ""))
        return 1, msgs
    md, csv_text = render(rows, min_relevance, known)
    open(os.path.join(folder, stem + ".md"), "w", encoding="utf-8", newline="\n").write(md)
    open(os.path.join(folder, stem + ".csv"), "w", encoding="utf-8", newline="").write(csv_text)
    msgs = [f"{len(rows)} notes -> {os.path.join(folder, stem + '.md')} and .csv"]
    if known is not None:
        hit = [f"{r['id']} ({known[norm_url(r['paper'])]})" for r in rows if norm_url(r["paper"]) in known]
        msgs.append(f"{len(hit)} of {len(rows)} notes are already known: " + ", ".join(hit) if hit else "none of the notes is a known paper")
    return 0, msgs


def set_result(folder, note_id, text):
    """write the `- result:` line of one note's header; (exit code, message). The table is made again by the caller."""
    text = (text or "").strip()
    note = os.path.join(folder, note_id, "note.md")
    if not FOLDER.match(note_id) or os.path.islink(os.path.join(folder, note_id)) or not os.path.isfile(note):
        return 1, f"no note.md for {note_id!r} in {folder}"
    if not RESULT.match(text):
        return 1, "result is not '-' or '<experiment id>: KEEP / CUT / TIE / INCONCLUSIVE / INVALID / MEASURED / planned / running [words]'"
    raw = open(note, encoding="utf-8", newline="").read()
    head, bad = header(raw)
    if bad:
        return 1, f"{note_id}: the header is not valid ({'; '.join(bad)}): fix the note first"
    new, n = re.subn(r"(?m)^- result:[ \t]*[^\r\n]*", lambda m: "- result: " + text, raw, count=1)
    if n != 1:
        return 1, f"{note_id}: no '- result:' line found"
    open(note, "w", encoding="utf-8", newline="").write(new)
    return 0, f"{note_id}: result = {text}"


def note_ids(folder):
    """ids of the paper subfolders of a topic that hold a note.md (odd names and links are left out: collect() names them)"""
    return [n for n in sorted(os.listdir(folder)) if not n.startswith((".", "_")) and FOLDER.match(n) and os.path.isdir(os.path.join(folder, n))
            and not os.path.islink(os.path.join(folder, n)) and os.path.isfile(os.path.join(folder, n, "note.md"))]


def topic_dirs(topics):
    """the topic folders below a topics folder (the hidden and the _ ones left out)"""
    return [os.path.join(topics, n) for n in sorted(os.listdir(topics)) if not n.startswith((".", "_")) and os.path.isdir(os.path.join(topics, n))] if os.path.isdir(topics) else []


def paper_of(note):
    """the paper link in the header of a note, '' if the note cannot be read"""
    try:
        return header(open(note, encoding="utf-8").read())[0].get("paper", "")
    except (OSError, ValueError):
        return ""


def set_result_all(folder, topics, note_id, text):
    """set_result for the named note and for every note on the same paper (normalised link) in the other topics of the problem: [(topic folder, exit code, message)]"""
    code, msg = set_result(folder, note_id, text)
    out = [(folder, code, msg)]
    if code:
        return out
    key = norm_url(paper_of(os.path.join(folder, note_id, "note.md")))
    for t in topic_dirs(topics):
        if os.path.realpath(t) == os.path.realpath(folder):
            continue
        for n in note_ids(t):
            p = paper_of(os.path.join(t, n, "note.md"))
            if p and norm_url(p) == key:
                out.append((t, *set_result(t, n, text)))
    return out


def stale(folder, rows):
    """table.md and table.csv are missing or do not hold what the notes give now (the column `known` aside): [reasons]"""
    expected = dict(zip(("md", "csv"), render(rows)))
    out = []
    for ext, want in expected.items():
        t = os.path.join(folder, f"table.{ext}")
        if not os.path.isfile(t):
            out.append(f"table.{ext} is missing: run the tool")
            continue
        with open(t, encoding="utf-8", newline="") as f:
            got = f.read()
        lines = got.split("\n")
        if ext == "md" and "| experiment result | known |" in got:
            lines = [re.sub(r" (known|yes|-) \|$", "", ln) if ln.startswith("| [") or ln.startswith("| title") else
                     (ln[:-4] if ln.startswith("|---") else ln) for ln in lines]
        elif ext == "csv" and lines[0].endswith(",known"):
            lines = [ln.rsplit(",", 1)[0] if ln else ln for ln in lines]
        if "\n".join(lines) != want:
            out.append(f"table.{ext} does not hold what the notes give now: run the tool again")
    return out


def prose(text):
    """markdown without its code spans (a `[x](y)` inside backticks is text, not a link)"""
    return re.sub(r"`[^`\n]*`", "", text)


def broken_links(folder):
    """the relative markdown links of overview.md and table.md that point at nothing; overview.md must exist and link a note: [reasons]"""
    out = []
    for name in ("overview.md", "table.md"):
        p = os.path.join(folder, name)
        if not os.path.isfile(p):
            if name == "overview.md":  # a missing table.md is named by stale()
                out.append("overview.md is missing")
            continue
        with open(p, encoding="utf-8", errors="replace") as f:
            targets = dict.fromkeys(re.findall(r"(?<!\\)\]\(\s*([^)\s]+)", prose(f.read())))  # an escaped \]( in a table cell is text
        if name == "overview.md" and not any(re.split(r"[#?]", t)[0].endswith("note.md") for t in targets):
            out.append("overview.md links no note")
        for t in targets:
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", t) or t.startswith("#"):  # http(s), mailto, ... and a page-internal anchor
                continue
            if not os.path.exists(os.path.join(folder, unquote(re.split(r"[#?]", t)[0]))):
                out.append(f"{name}: the link {t} does not resolve")
    return out


def rejected(folder):
    """the names under _rejected/ (a paper id, or <id>.<ext>)"""
    gone = os.path.join(folder, "_rejected")
    return os.listdir(gone) if os.path.isdir(gone) else []


def is_rejected(note_id, gone):
    return any(g == note_id or g.startswith(note_id + ".") for g in gone)


def check_md_problems(folder):
    """check.md (the spot check): one bullet per claim with its verdict and note id, a confirmed claim at least, no failed claim of a paper
    still in the topic, and no note written after it: [reasons]"""
    p = os.path.join(folder, "check.md")
    if not os.path.isfile(p):
        return ["check.md is missing (the spot check of the procedure)"]
    notes, gone, out, confirmed = note_ids(folder), rejected(folder), [], 0
    with open(p, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    for i, ln in enumerate(lines, 1):
        if not ln.lstrip().startswith(("- ", "* ")):
            if re.search(r"\b(contradicted|unreadable)\b", ln, re.I) and not ln.lstrip().startswith("#"):
                out.append(f"check.md line {i}: a failed claim outside a claim line: {ln.strip()[:80]}")
            continue
        m = re.match(r"\s*[-*] (confirmed|contradicted|unreadable): `?([A-Za-z0-9][A-Za-z0-9._-]*?)`?:", ln)
        if not m:
            out.append(f"check.md line {i}: not '- confirmed|contradicted|unreadable: <note id>: <claim>': {ln.strip()[:80]}")
            continue
        verdict, note_id = m.groups()
        if is_rejected(note_id, gone):
            continue
        if note_id not in notes:
            out.append(f"check.md line {i}: {note_id} is no note of this topic")
        elif verdict != "confirmed":
            out.append(f"check.md line {i}: {ln.strip()[:100]}")
        else:
            confirmed += 1
    if not confirmed:
        out.append("check.md holds no confirmed claim of a note of this topic")
    newer = [n for n in notes if os.path.getmtime(os.path.join(folder, n, "note.md")) > os.path.getmtime(p) + 2]
    if newer:
        out.append(f"written after the check: {', '.join(newer)} (check again)")
    return out


def candidate_ids(path, skip_known=False):
    """the ids (first column) of the data rows of every markdown table in a candidates file whose first column is headed `id`; skip_known: not the
    rows whose column `known?` (the header cell starts with 'known') says yes - the paper is already a note of the problem and needs no folder here"""
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    sep = lambda s: s.lstrip().startswith("|") and "-" in s and not set(s) - set("|-: \t")  # noqa: E731
    ids, known_col, listed = [], None, False
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("|") or sep(ln):
            continue
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", ln.strip()[1:])]
        if i + 1 < len(lines) and sep(lines[i + 1]):                       # a header row
            known_col = next((k for k, c in enumerate(cells) if c.lower().startswith("known")), None)
            listed = cells[0].strip("*` ").lower() == "id"                  # other tables of the file are not candidate lists
            continue
        if not listed:
            continue
        if skip_known and known_col is not None and known_col < len(cells) and cells[known_col].lower().startswith("yes"):
            continue
        ids.append(re.sub(r"^\[([^\]]*)\]\([^)]*\)$", r"\1", cells[0]).strip("`* "))
    return ids


def candidate_gaps(folder):
    """ids of the candidate list that have no folder and are not under _rejected/, or no candidate at all: [reasons]. The list is candidates.md (the
    merge); candidates_<subtopic>.md files are the inputs of the merge and count only while there is no candidates.md. Rows marked known (column
    `known?`) need no folder"""
    gone = rejected(folder)
    out = []
    names = sorted(n for n in os.listdir(folder) if re.fullmatch(r"candidates.*\.md", n))
    if "candidates.md" in names:
        names = ["candidates.md"]
    if not any(candidate_ids(os.path.join(folder, n)) for n in names):
        out.append("no candidate list (candidates.md with a table whose first column is `id`)")
    for n in names:
        for i in candidate_ids(os.path.join(folder, n), skip_known=True):
            if not FOLDER.match(i):
                out.append(f"{n}: the id {i!r} is not folder-safe (letters, digits, . _ -, a letter or digit first)")
            elif not os.path.isdir(os.path.join(folder, i)) and not is_rejected(i, gone):
                out.append(f"{n}: the id {i} has no folder and is not under _rejected/")
    return out


def hidden_notes(folder):
    """note.md files in folders that start with _ or . (the table skips them) anywhere below the topic, except _rejected itself: [reasons]"""
    return [f"{os.path.relpath(d, folder)}: a note in a folder that starts with _ or . is skipped by the table (rename the folder, or move a rejected paper under _rejected/)"
            for d, _, files in os.walk(folder) if "note.md" in files and d != folder and os.path.basename(d).startswith(("_", ".")) and os.path.basename(d) != "_rejected"]


def duplicate_papers(topics):
    """warnings for a paper (normalised link) that is a note in more than one topic of the problem"""
    seen = {}
    for t in topic_dirs(topics):
        for n in note_ids(t):
            if paper_of(os.path.join(t, n, "note.md")):
                seen.setdefault(norm_url(paper_of(os.path.join(t, n, "note.md"))), []).append(f"{os.path.basename(t)}/{n}")
    return [f"the same paper is a note in several topics: {', '.join(v)}" for v in seen.values() if len({x.split('/')[0] for x in v}) > 1]


def check(folder, topics=None):
    """the done state of a topic, nothing written; (exit code, messages). `topics` (the topics folder of the problem) adds the warning for a paper in several topics."""
    rows, errors = collect(folder)
    bad = [f"{w}: {'; '.join(why)}" for w, why in errors]
    bad += ([] if errors else stale(folder, rows)) + broken_links(folder) + check_md_problems(folder) + candidate_gaps(folder) + hidden_notes(folder)
    msgs = ["problem: " + b for b in bad] + ["warning: " + w for w in (duplicate_papers(topics) if topics else [])]
    return (1 if bad else 0), msgs + [f"{len(bad)} problem(s)" if bad else f"ok: {folder} is in its done state"]


def topics_root(folder):
    """the topics folder that holds this topic (problems/<p>/literature/topics), None for a topic that lies elsewhere"""
    parent = os.path.dirname(os.path.abspath(folder))
    return parent if os.path.basename(parent) == "topics" else None


def resolve(arg, repo=REPO):
    """the topic folder of an argument: a folder as given, else <problem>/<slug> -> problems/<problem>/literature/topics/<slug>, else the argument"""
    if os.path.isdir(arg):
        return arg
    m = re.fullmatch(r"([a-z][a-z0-9_]*)/([A-Za-z0-9][A-Za-z0-9._-]*)", arg)
    if m:
        topic = os.path.join(repo, "problems", m.group(1), "literature", "topics", m.group(2))
        if os.path.isdir(topic):
            return topic
    return arg


def _st_note(topic, name, body):
    os.makedirs(os.path.join(topic, name), exist_ok=True)
    open(os.path.join(topic, name, "note.md"), "w", encoding="utf-8", newline="").write(body)


def _st_resolve(tmp):
    """a folder is taken as given; <problem>/<slug> names a problem's topic; the topics folder is the one that holds the topic"""
    repo, local = os.path.join(tmp, "repo"), os.path.join(tmp, "local")
    topic = os.path.join(repo, "problems", "prob", "literature", "topics", "a-slug")
    os.makedirs(topic)
    os.makedirs(local)
    assert os.path.normpath(resolve("prob/a-slug", repo)) == os.path.normpath(topic)
    assert resolve(local, repo) == local and resolve("prob/none", repo) == "prob/none" and resolve("../x", repo) == "../x"
    global REPO
    real, REPO = REPO, repo
    try:
        assert os.path.normpath(resolve("prob/a-slug", REPO)) == os.path.normpath(topic)
    finally:
        REPO = real
    assert os.path.isfile(os.path.join(REPO, "autoresearch", "literature", "topic_table.py")), "REPO is the repository"
    assert topics_root(topic) == os.path.dirname(topic) and topics_root(local) is None


def patch_note(topic, name, old, new):
    p = os.path.join(topic, name, "note.md")
    with open(p, encoding="utf-8") as f:
        text = f.read()
    with open(p, "w", encoding="utf-8") as f:
        f.write(text.replace(old, new))


def _st_check(ok, tmp):
    """--check: a topic in its done state passes, and each rule is broken once"""
    import shutil
    import time
    top = os.path.join(tmp, "topics")
    good = os.path.join(top, "a")
    for n, t in (("xx3", "Aardvark"), ("yy2", "Beta [b](c) | x")):  # the table escapes this title: its \](c) is text, not a link to check
        _st_note(good, n, ok.format(n=n, t=t, r="2", c="-"))
    _st_note(os.path.join(good, "_rejected"), "rej1", ok.format(n="rej1", t="Rejected", r="1", c="-"))
    head = "# Candidates\n\n| id | title | paper |\n|---|---|---|\n"
    open(os.path.join(good, "candidates.md"), "w").write(head + "| xx3 | Aardvark | https://x.org/xx3 |\n| [yy2](yy2/note.md) | Beta | https://x.org/yy2 |\n| `rej1` | Rejected | https://x.org/rej1 |\n")
    open(os.path.join(good, "candidates_sub.md"), "w").write(head + "| xx3 | Aardvark | https://x.org/xx3 |\n")
    open(os.path.join(good, "a b.md"), "w").write("x")
    assert run(good)[0] == 0
    open(os.path.join(good, "overview.md"), "w").write("# O\n\n[t](table.md) [n](xx3/note.md) [w](https://x.org) [anchor](#top) [f](table.md#x) [sp](a%20b.md) `W[c](x)`\n")
    open(os.path.join(good, "check.md"), "w").write("# Spot check\n\n- confirmed: xx3: does a (https://x.org/xx3)\n- unreadable: rej1: a claim of a rejected paper (https://x.org/rej1)\n")
    code, msgs = check(good)
    assert code == 0 and msgs == [f"ok: {good} is in its done state"], msgs

    def variant(name, change, *want):
        c = os.path.join(tmp, "v_" + name)
        shutil.copytree(good, c)
        change(c)
        code, msgs = check(c)
        assert code == 1 and all(any(w in m for m in msgs) for w in want), (name, want, msgs)
        shutil.rmtree(c)

    def write(name, text, mode="a"):
        return lambda c: open(os.path.join(c, name), mode).write(text)

    def touch(name, ahead):
        t = time.time() + ahead
        return lambda c: os.utime(os.path.join(c, name), (t, t))
    variant("stale_table", write("table.md", "| edited |\n"), "table.md does not hold what the notes give now")
    variant("stale_csv", write("yy2/note.md", "", "a") if False else lambda c: patch_note(c, "yy2", "- relevance: 2", "- relevance: 1"), "table.md does not hold", "table.csv does not hold")
    variant("newer_note", touch("yy2/note.md", 100), "written after the check: yy2")
    variant("no_csv", lambda c: os.remove(os.path.join(c, "table.csv")), "table.csv is missing")
    variant("no_md", lambda c: os.remove(os.path.join(c, "table.md")), "table.md is missing")
    variant("overview_link", write("overview.md", "[x](nothere.md)\n"), "overview.md: the link nothere.md does not resolve")
    variant("no_overview", lambda c: os.remove(os.path.join(c, "overview.md")), "overview.md is missing")
    variant("empty_overview", write("overview.md", "# O\n", "w"), "overview.md links no note")
    variant("empty_check", write("check.md", "", "w"), "no confirmed claim")
    variant("free_text_check", write("check.md", "- I could not open any source\n", "w"), "not '- confirmed|contradicted|unreadable", "no confirmed claim")
    variant("unknown_note", write("check.md", "- confirmed: zz0: x (u)\n"), "zz0 is no note of this topic")
    variant("no_candidates", lambda c: [os.remove(os.path.join(c, n)) for n in ("candidates.md", "candidates_sub.md")], "no candidate list")
    variant("table_link", lambda c: shutil.rmtree(os.path.join(c, "yy2")), "table.md: the link yy2/note.md does not resolve")  # the escaped brackets of a title do not hide a link either
    variant("no_check", lambda c: os.remove(os.path.join(c, "check.md")), "check.md is missing")
    variant("contradicted", write("check.md", "- contradicted: yy2: claim (https://x.org)\n"), "check.md line 5: - contradicted: yy2")
    variant("unreadable", write("check.md", "- unreadable: yy2: claim (https://x.org)\n"), "check.md line 5")
    variant("outside", write("check.md", "yy2 is contradicted by its page\n"), "a failed claim outside a claim line")
    variant("candidate_gone", write("candidates.md", "| gone9 | Gone | https://x.org/gone9 |\n"), "candidates.md: the id gone9 has no folder")
    c = os.path.join(tmp, "v_known")                                            # a row marked known needs no folder; candidates_<sub>.md is an input of the merge, not checked once candidates.md exists
    shutil.copytree(good, c)
    open(os.path.join(c, "candidates.md"), "a").write("\n| id | title | paper | known? |\n|---|---|---|---|\n| zz9 | Z | https://x.org/zz9 | yes |\n| zy8 | Y | https://x.org/zy8 | no |\n")
    open(os.path.join(c, "candidates_sub.md"), "a").write("| gone8 | Gone | https://x.org/gone8 |\n")
    run(c)
    code, msgs = check(c)
    assert code == 1 and any("candidates.md: the id zy8 has no folder" in m for m in msgs) and not any("zz9" in m or "gone8" in m for m in msgs), msgs
    shutil.rmtree(c)
    variant("candidate_unsafe", write("candidates.md", "| bad id | Bad | https://x.org/bad |\n"), "not folder-safe")
    variant("no_note_md", lambda c: os.makedirs(os.path.join(c, "empty1")), "empty1: no note.md")
    variant("bad_header", write("yy2/note.md", "# x\n- title: only\n", "w"), "header must be exactly")

    def hidden(c):
        for p in ("_work", os.path.join("_rejected", "_x"), ".hid"):
            _st_note(c, p, "# h\n")
    variant("hidden", hidden, "_work: a note in a folder", os.path.join("_rejected", "_x") + ": a note", ".hid: a note")
    c = os.path.join(tmp, "v_reject")                              # the rejection path ends in the done state: a failed claim of a rejected paper
    shutil.copytree(good, c)
    open(os.path.join(c, "check.md"), "a").write("- contradicted: yy2: claim (https://x.org)\n")
    shutil.move(os.path.join(c, "yy2"), os.path.join(c, "_rejected", "yy2"))
    with open(os.path.join(c, "candidates.md")) as f:
        kept = [ln for ln in f.read().split("\n") if "yy2" not in ln]
    with open(os.path.join(c, "candidates.md"), "w") as f:
        f.write("\n".join(kept) + "\n| known id | what |\n|---|---|\n| TP9 | another table is no candidate list |\n")
    with open(os.path.join(c, "overview.md")) as f:
        over = f.read()
    with open(os.path.join(c, "overview.md"), "w") as f:
        f.write(over.replace("[sp](a%20b.md)", ""))
    assert run(c)[0] == 0 and check(c) == (0, [f"ok: {c} is in its done state"]), check(c)
    run(c, 0, {"x.org/xx3": "K1"})                                  # a table with the column `known` still holds what the notes give
    assert check(c)[0] == 0, check(c)
    shutil.rmtree(c)
    # a paper that is a note in two topics: a warning only with the topics folder, never an error
    _st_note(os.path.join(top, "b"), "zz9", ok.format(n="zz9", t="Same", r="1", c="-").replace("x.org/zz9", "x.org/xx3#frag"))
    assert check(good) == (0, [f"ok: {good} is in its done state"]), "no warning without --all-topics"
    code, msgs = check(good, top)
    assert code == 0 and [m for m in msgs if m.startswith("warning")] == ["warning: the same paper is a note in several topics: a/xx3, b/zz9"], msgs  # one paper, one line


def _st_known(ok, tmp):
    """--known: the link column of a papers table is found (by name or by content), the table gets the column `known`, a table without links is refused"""
    d = os.path.join(tmp, "known")
    for n, paper in (("kn1", "http://www.arxiv.org/pdf/2401.01234.pdf"), ("kn2", "https://arxiv.org/abs/2402.00001v3"), ("kn3", "https://x.org/kn3")):
        _st_note(d, n, ok.format(n=n, t=n, r="1", c="-").replace("https://x.org/" + n, paper))

    def table(name, header, *rows):
        p = os.path.join(tmp, name)
        with open(p, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        return p
    named = table("papers.csv", ["id", "title", "url", "code"], ["P1", "Alpha", "https://arxiv.org/abs/2401.01234v2", "https://github.com/a/a"], ["2402.00001", "", "", ""],
                  ["SN01", "", "", "skip: ours"], ["P3", "Gamma", "https://x.org/other", ""])
    assert load_known(named) == {"arxiv:2401.01234": "P1", "arxiv:2402.00001": "2402.00001", "x.org/other": "P3"}, load_known(named)  # the arXiv id of a row without link counts
    assert load_known(table("p2.csv", ["id", "Paper Link", "code"], ["A", "https://x.org/kn3", "-"])) == {"x.org/kn3": "A"}, "a link-like name, any case"
    assert load_known(table("p3.csv", ["id", "where", "code"], ["A", "https://x.org/kn3", "-"], ["B", "https://x.org/q", "-"])) == {"x.org/kn3": "A", "x.org/q": "B"}, "no name, but a column of links"
    assert load_known(table("p6.csv", ["id", "code", "url"], ["A", "https://github.com/a/a", "https://x.org/kn3"])) == {"x.org/kn3": "A"}, "the link-like name beats a column of repository links"
    for bad in (table("p4.csv", ["id", "title"], ["A", "Alpha"]), table("p5.csv", ["id", "code"], ["A", "https://x.org/kn3"], ["B", "-"], ["C", "-"]), os.path.join(tmp, "none.csv")):
        try:
            load_known(bad)
            raise AssertionError("no column of links must be refused: " + bad)
        except ValueError as e:
            assert "no column that holds paper links" in str(e) or "cannot read" in str(e), e
    rows = collect(d)[0]
    plain_md, plain_csv = render(rows)
    assert "known" not in plain_md and plain_csv.splitlines()[0] == "id,title,summary,relevance,status,verdict,paper,code,result,note", "without --known the table is as before"
    md, csv_text = render(rows, 0, load_known(named))
    assert "| experiment result | known |" in md and csv_text.splitlines()[0].endswith(",note,known"), md
    marks = {ln.split("]")[0][3:]: ln.rsplit("|", 2)[-2].strip() for ln in md.splitlines() if ln.startswith("| [")}
    assert marks == {"kn1": "yes", "kn2": "yes", "kn3": "-"}, marks
    assert [ln.rsplit(",", 1)[-1] for ln in csv_text.splitlines()[1:]] == ["yes", "yes", "-"], csv_text
    code, msgs = run(d, 0, load_known(named))
    assert code == 0 and "2 of 3 notes are already known: kn1 (P1), kn2 (2402.00001)" in msgs[1], msgs
    catalogue = os.path.join(tmp, "CATALOGUE.md")                    # a catalogue or register: every link of a row counts, arXiv html too
    with open(catalogue, "w", encoding="utf-8") as f:
        f.write("# C\n\n| id | paper | original |\n|---|---|---|\n| **G1** | Gamma | [arXiv](https://arxiv.org/html/2402.00001v2) [repo](https://x.org/kn3) |\n")
    assert load_known(catalogue) == {"arxiv:2402.00001": "G1", "x.org/kn3": "G1"}, load_known(catalogue)
    assert norm_url("https://ar5iv.labs.arxiv.org/html/2401.01234") == norm_url("https://export.arxiv.org/abs/2401.01234v1") == "arxiv:2401.01234"


def _st_flows(ok, tmp):
    """the command line: --set-result with and without --all, --check, --all-topics, --known, and the exit code 2 of a wrong call"""
    top = os.path.join(tmp, "flows", "literature", "topics")
    for t, n, paper in (("a", "n1", "https://x.org/shared"), ("b", "n2", "https://X.org/shared/#sec"), ("b", "n4", "https://x.org/z"), ("c", "n3", "https://x.org/y")):
        _st_note(os.path.join(top, t), n, ok.format(n=n, t=n.upper(), r="1", c="-").replace("https://x.org/" + n, paper))
    for t in "abc":
        assert run(os.path.join(top, t))[0] == 0

    def call(*argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                main(list(argv))
                code = 0
            except SystemExit as e:
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        return code, buf.getvalue()

    def read(t, *p):
        return open(os.path.join(top, t, *p), encoding="utf-8").read()
    a = os.path.join(top, "a")
    code, out = call(a, "--set-result", "n1", "E1: KEEP 0.1")
    assert code == 0 and "- result: E1: KEEP 0.1" in read("a", "n1", "note.md") and "- result: -" in read("b", "n2", "note.md"), "without --all only the named note changes"
    code, out = call(a, "--set-result", "n1", "E2: CUT 0.2", "--all")
    assert code == 0 and out.count("result = E2: CUT 0.2") == 2 and "a/n1: result" in out and "b/n2: result" in out, out
    assert "- result: E2: CUT 0.2" in read("b", "n2", "note.md") and "- result: -" in read("b", "n4", "note.md") and "- result: -" in read("c", "n3", "note.md"), "same paper only (the # fragment, the case of the host do not matter)"
    assert "E2: CUT 0.2" in read("a", "table.md") and "E2: CUT 0.2" in read("b", "table.md") and "E2" not in read("c", "table.md"), "the table of each touched topic is made again, the others are left"
    assert call(a, "--set-result", "nope", "E3: KEEP", "--all")[0] == 1 and "E3" not in read("b", "n2", "note.md"), "a wrong note id changes nothing"
    assert call(a, "--all")[0] == 2 and call(a, "--all-topics")[0] == 2 and call(a, "--check", "--known", "x.csv")[0] == 2, "options that do not go together"
    csv_no_link = os.path.join(tmp, "nolink.csv")
    open(csv_no_link, "w").write("id,title\nA,B\n")
    assert call(a, "--known", csv_no_link)[0] == 2, "a papers table without links is refused"
    papers = os.path.join(tmp, "papers.csv")
    open(papers, "w").write("id,title,url\nP9,Shared,https://x.org/shared/\n")
    code, out = call(a, "--known", papers)
    assert code == 0 and "1 of 1 notes are already known: n1 (P9)" in out and "| known |" in read("a", "table.md"), out
    code, out = call(a, "--set-result", "n1", "E4: TIE", "--min-relevance", "3")
    assert code == 0 and "E4: TIE" in read("a", "table.md") and os.path.isfile(os.path.join(a, "table_min3.md")), "the full table holds the new result"
    code, out = call(a, "--check", "--all-topics")
    assert code == 1 and "warning: the same paper is a note in several topics: a/n1, b/n2" in out and "problem: overview.md is missing" in out, out
    code, out = call(a, "--check")
    assert code == 1 and "warning" not in out and "check.md is missing" in out, out


def selftest():
    import shutil
    import tempfile
    d = tempfile.mkdtemp(prefix="ar_topic_")
    try:
        def note(name, body):
            os.makedirs(os.path.join(d, name), exist_ok=True)
            open(os.path.join(d, name, "note.md"), "w", encoding="utf-8", newline="").write(body)

        ok = ("# {n} t\n\n<!-- a comment line is fine -->\n- title: {t}\n- relevance: {r}\n- summary: It does a. It matters because b.\n- status: checked\n"
              "- paper: https://x.org/{n}\n- code: {c}\n- verdict: use\n- result: -\n\n## More\n- code: https://body.example/ignored\n")
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
            "badval": ("# d\n- title: D\n- relevance: 5\n- summary: s\n- status: guess\n- paper: ftp://x\n- code: nowhere\n- verdict: maybe\n- result: KEEP\n", ["result is not", "relevance is not 0-3", "paper is not", "code is not", "verdict is not", "status is not"]),
            "short": ("# e\n- title: Eps\n", ["header must be exactly"]),
            "order": ("# o\n- relevance: 1\n- title: T\n- summary: s\n- status: checked\n- paper: https://x.org/o\n- code: -\n- verdict: use\n- result: -\n", ["header must be exactly"]),
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
        for stale_name in ("table.md", "table.csv", "table_min2.md", "table_min2.csv", "table_notes.md"):  # the last one is not a table of this tool
            text = "# Table of the topic\nold" if stale_name.endswith(".md") else "id,title,summary,relevance,status,old"
            open(os.path.join(d, stale_name), "w").write(text)
        code, msgs = run(d)
        assert code == 1 and "removed the old table.md, table.csv, table_min2.md, table_min2.csv" in msgs[-1], msgs
        assert sorted(n for n in os.listdir(d) if n.startswith("table")) == ["table_notes.md"], "every table of the tool goes, other files stay"
        os.remove(os.path.join(d, "table_notes.md"))
        for own in ("table.md", "table.csv"):                              # a file of that name the tool did not write stays
            open(os.path.join(d, own), "w").write("kept by hand")
        assert run(d)[0] == 1 and open(os.path.join(d, "table.md")).read() == "kept by hand" and os.path.isfile(os.path.join(d, "table.csv"))
        os.remove(os.path.join(d, "table.md"))
        os.remove(os.path.join(d, "table.csv"))
        for name in list(cases) + ["nonote", "bad name", "_work"]:
            shutil.rmtree(os.path.join(d, name))
        code, msgs = run(d)
        assert code == 0 and os.path.isfile(os.path.join(d, "table.md")) and os.path.isfile(os.path.join(d, "table.csv")), msgs
        code, msgs = run(d, 2)
        assert code == 0 and os.path.isfile(os.path.join(d, "table_min2.md")) and "Aardvark" in open(os.path.join(d, "table.md")).read() and "Alpha" in open(os.path.join(d, "table.md")).read() \
            and "Alpha" not in open(os.path.join(d, "table_min2.md")).read(), "a filtered table has its own file and table.md stays complete"
        # the result line: set in place (CRLF kept), shown in the table, refused for a bad text, a bad id or an invalid note
        note("crlf1", ok.format(n="crlf1", t="Crlf", r="1", c="-").replace("\n", "\r\n"))
        before = open(os.path.join(d, "crlf1", "note.md"), "rb").read()
        assert set_result(d, "crlf1", "E283: CUT lane IoU -0.005")[0] == 0 and set_result(d, "xx3", "T24: running")[0] == 0
        after = open(os.path.join(d, "crlf1", "note.md"), "rb").read()
        assert after.count(b"\r\n") == before.count(b"\r\n") and b"- result: E283: CUT lane IoU -0.005\r\n" in after and after.count(b"- result:") == 1, after
        assert run(d)[0] == 0 and "E283: CUT lane IoU -0.005" in open(os.path.join(d, "table.md")).read() and "| T24: running |" in open(os.path.join(d, "table.md")).read()
        assert "result" in open(os.path.join(d, "table.csv")).read().splitlines()[0]
        assert all(set_result(d, "xx3", f"E1: {v}")[0] == 0 for v in ("INVALID", "MEASURED", "planned"))
        for bad_id, bad_text in (("xx3", "KEEP"), ("xx3", "E1: MAYBE"), ("nope", "-"), ("../x", "-"), ("xx3", "E1: KEEP " + "x" * 200)):
            assert set_result(d, bad_id, bad_text)[0] == 1, (bad_id, bad_text)
        assert set_result(d, "xx3", "-")[0] == 0 and "- result: -" in open(os.path.join(d, "xx3", "note.md")).read(), "back to empty"
        # now break one note: set_result refuses to write into an invalid header
        note("hdr", "# h\n- title: only\n")
        assert set_result(d, "hdr", "-")[0] == 1
        shutil.rmtree(os.path.join(d, "hdr"))
        shutil.rmtree(os.path.join(d, "crlf1"))
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
        tmp = tempfile.mkdtemp(prefix="ar_topic_more_")
        try:
            _st_resolve(tmp)
            _st_check(ok, tmp)
            _st_known(ok, tmp)
            _st_flows(ok, tmp)
        finally:
            shutil.rmtree(tmp)
    finally:
        shutil.rmtree(d)
    print("topic_table selftest passed")


def main(argv):
    if argv[:1] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topic", help="the topic folder, or <problem>/<slug>")
    ap.add_argument("--min-relevance", type=int, default=0, choices=[0, 1, 2, 3])
    ap.add_argument("--set-result", nargs=2, metavar=("NOTE_ID", "TEXT"), help="fill the result line of one note, then make the table again")
    ap.add_argument("--all", action="store_true", help="with --set-result: also the notes on the same paper in the other topics of that folder (their tables are made again)")
    ap.add_argument("--known", metavar="FILE", nargs="+", help="what the problem holds (papers.csv, CATALOGUE.md, register.md): list the notes whose paper is in it and add the column `known`")
    ap.add_argument("--check", action="store_true", help="the done state of the topic (writes nothing, exit 1 on any problem)")
    ap.add_argument("--all-topics", action="store_true", help="with --check: warn about a paper that is a note in more than one topic of that folder")
    a = ap.parse_args(argv)
    if a.all and not a.set_result:
        ap.error("--all goes with --set-result")
    if a.all_topics and not a.check:
        ap.error("--all-topics goes with --check")
    if a.check and (a.set_result or a.known):
        ap.error("--check writes nothing: it takes no --set-result and no --known")
    known = None
    if a.known:
        try:
            known = {}
            for path in a.known:
                for link, rid in load_known(path).items():
                    known.setdefault(link, rid)
        except ValueError as e:
            ap.error(str(e))  # exit 2
    folder = resolve(a.topic)
    if not os.path.isdir(folder):
        sys.exit(f"{a.topic} is not a topic folder or <problem>/<slug>")
    topics = topics_root(folder)
    if (a.all or a.all_topics) and not topics:
        ap.error("--all / --all-topics need a topic that lies in a topics folder")
    if a.check:
        code, msgs = check(folder, topics if a.all_topics else None)
        print("\n".join(msgs))
        sys.exit(code)
    touched, failed = [folder], 0
    if a.set_result:
        results = set_result_all(folder, topics, *a.set_result) if a.all else [(folder, *set_result(folder, *a.set_result))]
        for t, code, msg in results:
            print(f"{os.path.basename(os.path.abspath(t))}/{msg}" if a.all else msg)
        if results[0][1]:  # the named note itself failed: nothing was written
            sys.exit(1)
        failed = sum(bool(c) for _, c, _ in results)
        touched = list(dict.fromkeys(t for t, c, _ in results if not c))  # the topics whose notes changed, the named one first
    for t in touched:
        for level in dict.fromkeys([0, a.min_relevance] if a.set_result else [a.min_relevance]):   # a new result reaches the full table too
            code, msgs = run(t, level, known)
            print("\n".join(msgs))
            failed += bool(code)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
