"""The TEST gate as one small module: does a row of the holdout log open the holdout split for a run id? Standard library only, no imports of the other tools at load time, so a
project's own evaluator can load it by path (importlib) without a changed sys.path or a clash of module names (the lane has its own `hold.py`).

    run.py          the launcher refuses `--split test` unless test_log_refusal(...) is empty
    evaluate.py     the lane's evaluator asks test_log_opens(...) before it scores the TEST split

The rule (hard rule 1 of METHOD.md: TEST only with a logged row under the person's recorded OK). Positive forms only: what is not written in one of them opens nothing.
  * the run id is a whole token in the 'who / which run' cell (the fourth) of a TABLE ROW of the log (a line that starts with |; a free-text line opens nothing, a mention as a
    comparison in another cell opens nothing, and the id `T.1` is not in `T.10` or `T.1.b`; a log with another layout than six columns: anywhere in the row);
  * the LAST row of that id decides, and an id that has a USED row (DONE, already used) anywhere is closed for good: log a new id in a new row;
  * that row carries the person's OK in its last cell (the note): the cell STARTS with OK (optionally `the person's OK`, `Mirko's OK`, `user's OK`) and holds no word of a condition or a
    negation (not, without, never, missing, pending, cancelled, revoked, wait, ask, need, expected, later, until, if, when, `for VAL`, a question mark); or the row cites a GRANT of a
    decision that grants a TEST check: `granted by D28` / `authorised|allowed|approved by|under D28`, or `D28 (Mirko 2026-10-31: "One TEST check")` (a date and a quotation in the
    parenthesis). A grant counts only when the project's decisions record calls the decision decided (a `Decided: <date>` line in its file) and the file says that it grants a TEST
    check (`one TEST check`, `TEST ... granted / allowed`, `grants ... TEST`; a design called D7 or a decision that only mentions TEST grants nothing);
  * the row does not say it is not approved (CLOSED_ROW: not approved, refused, denied, withdrawn, awaiting, no OK, OK?, requested, not yet answered, OK pending ...).
`grants` is the set of decision ids that grant a TEST check (grant_ids()); the default is the empty set: without a decisions record only the OK of the person opens.

    python3 testgate.py --selftest
"""
import importlib.util
import os
import re
import sys
import tempfile

CLOSED_ROW = re.compile(
    r"\b(?:not\s+approved|refused|denied|withdrawn|awaiting|waiting\s+for|no\s+OK|OK\s*[?？]|requested(?!\s+by)|to\s+be\s+(?:approved|answered|confirmed)|"
    r"not\s+yet\s+(?:answered|given|granted|approved)|(?:OK|approval|answer)\s+pending|pending\s+(?:OK|approval|answer)|pending[^|]{0,30}\b(?:yes|OK|approval|answer))(?!\w)",
    re.I)   # a holdout log row that says the run is not approved (a bare "pending" or "not yet" in the numbers cell is ordinary wording)
OK_NOTE = re.compile(r"^\s*(?:(?:the\s+)?(?:person|user|Mirko)(?:'s|’s)?\s+)?OK\b")            # the note cell starts with the person's OK ...
NOT_OK_NOTE = re.compile(r"\b(?:not|without|never|missing|pending|cancel\w*|revok\w*|wait\w*|ask\w*|need\w*|expected|later|tomorrow|soon|until|if|when|requested|"
                         r"for\s+VAL)\b|[?？]", re.I)   # ... and holds no condition or negation (a legitimate note that uses one of these words is worded differently: the gate fails closed)
GRANT_D = re.compile(r"\b(?:granted|authori[sz]ed|allowed|approved)\s+(?:by|under)\s+(D\d+)\b|(?<![\w.])(D\d+)\s*\((?=[^)]*\d{4}-\d{2}-\d{2}[^)]*[\"“”])", re.I)
GRANT_STMT = re.compile(r"\bone\s+TEST\s+check\b|\bTEST\s+(?:run|check|use|split)s?\s+(?:is|are|may\s+be)?\s*(?:granted|allowed|approved|OK)\b|\bgrant\w*\s+(?:\w+\s+){0,3}TEST\b|"
                        r"\bTEST\b[^.\n]{0,40}\b(?:granted|allowed)\b", re.I)    # what a decision file says when it grants a TEST check (D28: "One TEST check")
USED_ROW = re.compile(r"\bDONE\b|\b(?i:already\s+used)\b")                            # the run was used: the log writes DONE ("used" alone is ordinary wording)
LINES = re.compile(r"\r\n|\r|\n")        # only these end a row (str.splitlines also splits at U+2028, U+0085 and the form feed: a closing phrase would leave its row)


def _load_guard_hook():
    """tools/guard_hook.py by path: its own `sys.path.insert` and `import project` are undone afterwards (the module's functions keep working)"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "guard_hook.py")
    saved, had = list(sys.path), "project" in sys.modules
    try:
        spec = importlib.util.spec_from_file_location("ar_guard_hook", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
        if not had:
            sys.modules.pop("project", None)
    return mod


def grant_ids(root, cfg):
    """the decision ids of the project's decisions FOLDER that are decided (guard_hook.decided: a `Decided: <date>` line) and whose file says that it grants a TEST check
    (GRANT_STMT): the ones a row may cite. A project without a decisions folder, or a record that cannot be read, grants nothing: only the OK of the person opens (the gate fails closed)"""
    try:
        gh = _load_guard_hook()
        folder = os.path.join(root, cfg.get("decisions", "experiment_execution/DECISIONS.md"))
        if not os.path.isdir(folder):
            return set()
        out = set()
        for d in gh.decided(root, cfg):
            try:
                with open(os.path.join(folder, d + ".md"), encoding="utf-8") as f:
                    if GRANT_STMT.search(f.read()):
                        out.add(d)
            except (OSError, ValueError):
                continue
        return out
    except (Exception, SystemExit):
        return set()


def _cells(row):
    return [c.strip() for c in row.strip().strip("|").split("|")]


def test_log_refusal(log_text, run_id, grants=frozenset()):
    """'' when the holdout log opens the holdout split for this run id, else why not (the rule in the head of this file)"""
    tok = re.compile(r"(?<![\w.-])" + re.escape(run_id) + r"(?![\w-]|\.\w)")
    rows = []
    for ln in LINES.split(log_text):
        if not ln.lstrip().startswith("|"):
            continue
        cells = _cells(ln)
        if tok.search(cells[3] if len(cells) >= 6 else ln):
            rows.append(ln)
    if not rows:
        return "is not named in the 'who / which run' cell of a table row of the log"
    if any(USED_ROW.search(r) for r in rows):
        return "was already used according to its row in the log (DONE / already used): log a NEW id in a new row"
    last = rows[-1]                                         # the log is appended to: the newest row of the id decides
    if CLOSED_ROW.search(last):
        return "is named in the log, but its row says it is not approved (not approved / refused / denied / withdrawn / awaiting / no OK / requested)"
    note = _cells(last)[-1]
    if OK_NOTE.search(note) and not NOT_OK_NOTE.search(note):
        return ""
    cited = {a or b for a, b in GRANT_D.findall(last)}
    if cited & set(grants):
        return ""
    if cited:
        return (f"is named in the log, but its row cites {', '.join(sorted(cited))} and none of them is a decided decision that grants a TEST check (a 'Decided: <date>' line and "
                "'one TEST check' in its file of the decisions folder): the person's OK in the note cell, or such a decision")
    return "is named in the log, but its row carries no OK of the person (its note cell starts with OK, or `granted by D<n>` of a decision that grants a TEST check)"


def test_log_opens(log_text, run_id, grants=frozenset()):
    """does the holdout log open the holdout split for this run id? (test_log_refusal)"""
    return not test_log_refusal(log_text, run_id, grants)


def _selftest():
    head = "| date | loop | what was evaluated | who / which run | numbers | note |\n|---|---|---|---|---|---|\n"

    def row(rid, note, what="design", num="-", rid_cell=None):
        return f"| d | 1 | {what} | {rid_cell or 'run id ' + rid} | {num} | {note} |\n"
    log = head + row("E1.ok", "OK of 2026-11-10") + row("E1.po", "the person's OK of the loop prompt") + row("E1.mk", "Mirko's OK, once") \
        + row("E1.g", "granted by D28 (the person: one TEST check)") + row("E1.p", "-", what='D28 (Mirko 2026-10-31: "One TEST check")') \
        + row("E1.a", "authorised by D28") + row("E1.two", "pending") + row("E1.two", "OK of 2026-11-11") + row("E1.back", "OK") + row("E1.back", "refused later") \
        + row("E1.use", "OK", num="DONE") + row("E1.use2", "OK") + row("E1.use2", "DONE")
    dec = {"D28"}
    for rid in ("E1.ok", "E1.po", "E1.mk", "E1.g", "E1.p", "E1.a", "E1.two"):
        assert test_log_opens(log, rid, dec), rid
    for rid in ("E1.back", "E1.use", "E1.use2", "E1", "E1.none"):
        assert not test_log_opens(log, rid, dec), rid          # the newest row decides; a USED row closes the id for good
    assert test_log_opens(log, "E1.ok") and not test_log_opens(log, "E1.g") and not test_log_opens(log, "E1.g", set()), "the default grants nothing: only the OK of the person opens"
    assert "NEW id" in test_log_refusal(log, "E1.use", dec) and "NEW id" in test_log_refusal(log, "E1.use2", dec)
    # an OK that is a condition, a negation or in the wrong place opens nothing (a deny-list of the critic's list, and the positive form that makes it unnecessary)
    notes = ["not OK", "without OK yet", "Mirko never said OK", "OK for VAL only", "OK missing", "OK not given", "needs OK", "wait for OK", "waiting on Mirko, OK later", "ask for OK",
             "run if Mirko says OK", "OK expected tomorrow", "OK, cancelled", "OK revoked", "OK: pending", "OK, pending", "OK？", "no OK yet", "OK?", "requested, not yet answered",
             "pending Mirko's yes", "to be approved", "not yet given", "OK pending", "awaiting the OK", "refused", "-", "Mirko, in chat", "smoke test passed"]
    for n in notes:
        assert not test_log_opens(head + row("E2.x", n), "E2.x", dec), n
    assert not test_log_opens(head + row("E2.x", "planned", what="smoke OK"), "E2.x", dec) and not test_log_opens(head + row("E2.x", "planned", num="gate OK"), "E2.x", dec), "OK in another cell"
    assert not test_log_opens(head + row("E2.OK", "-"), "E2.OK", dec) and not test_log_opens(head + row("E2-OK", "-"), "E2-OK", dec), "OK inside the run id"
    assert not test_log_opens(head + row("E2.x", "OK not approved"), "E2.x", dec) and not test_log_opens(head + row("E2.x", "OK\x0cnot approved"), "E2.x", dec), "a row is not cut at U+2028 / form feed"
    # a decision opens a row only as a GRANT form, only when it is decided and says it grants a TEST check; a design called D7, a D<n> in the run id and a mention open nothing
    for n, what in (("granted by D7", "design"), ("design approved D4", "design"), ("allowed in D21", "design"), ("-", 'champion D7 ("rc1")'), ("-", 'vs control D28 ("x")'),
                    ("granted by D9999", "design"), ("granted in D28", "design"), ("D28", "design"), ("-", "design D28 (rc1, nothing granted)")):
        assert not test_log_opens(head + row("E2.x", n, what=what), "E2.x", {"D28", "D4"}), (n, what)       # D28 and D4 grant a TEST check here; D7 and D21 only mention it
    assert not test_log_opens(head + row("E2.D28", "-"), "E2.D28", dec) and not test_log_opens(head + row("E2.x", "-", what="design D28"), "E2.x", dec)
    assert "none of them is a decided decision that grants" in test_log_refusal(head + row("E2.x", "granted by D9999"), "E2.x", dec)
    assert not test_log_opens("Rule: E2.x may run.\n| x |\n", "E2.x", dec) and not test_log_opens("no table at all", "E2.x", dec), "free text and a log without rows"
    # grant_ids: a folder of decision files; decided = a 'Decided: <date>' line; the file says it grants a TEST check; no folder = nothing; loading changes neither sys.path nor sys.modules
    with tempfile.TemporaryDirectory() as root:
        os.makedirs(os.path.join(root, "dec"))
        for name, body in (("D28.md", "# D28\nDecided: 2026-10-31\nOne TEST check\n"), ("D7.md", "# D7\nDecided: 2026-10-25\nthe split, before its TEST check\n"),
                           ("D9.md", "# D9\nOne TEST check\nstill open\n"), ("D4.md", "# D4\nDecided: 2026-10-20\nTEST once is granted\n")):
            open(os.path.join(root, "dec", name), "w", encoding="utf-8").write(body)
        before = (list(sys.path), "project" in sys.modules, "hold" in sys.modules)
        assert grant_ids(root, {"decisions": "dec"}) == {"D28", "D4"}, "decided and granting a TEST check (D7 only mentions TEST, D9 is open)"
        assert (list(sys.path), "project" in sys.modules, "hold" in sys.modules) == before, "loading the gate changes neither sys.path nor sys.modules"
        assert grant_ids(root, {}) == set() and grant_ids("/no/such/root", {"decisions": "dec"}) == set(), "no decisions record grants nothing"
    return "testgate selftest passed"


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        print(_selftest())
    else:
        sys.exit(__doc__)
