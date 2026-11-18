"""Hold marker: a run id that must not be started again until the person says so (the hold marker; design rule 8 of ROADMAP.md). Standard library only.

    python3 autoresearch/tools/hold.py set ID --by "<chat>" --reason "..." [--words "<the person's words>"]
    python3 autoresearch/tools/hold.py clear ID --words "<the person's words>"       removes the marker, keeps it as HOLD.cleared.<date>, adds a ledger note
    python3 autoresearch/tools/hold.py show [ID]                                       the holds
    python3 autoresearch/tools/hold.py --selftest

The marker is the file <run records>/<ID>/HOLD, one JSON object: who, when, why, words. run.py writes it by itself when a run ends with exit 75 (a trainer that
was stopped on request) and refuses to start a run id that has one; `--after` chains stop at it (state `held`). Only the person's words clear it, never a chat's
own judgement: `clear` needs them, quoted, and the ledger row of the run gets a note with them. A relayed instruction is written as such (--words "relayed by
<chat>: ...") and the person's own word is still needed to clear it. This is hard rule 6 (the person's decisions are acted on only when recorded) for runs.
"""
import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

NAME = "HOLD"


def path(d):
    return os.path.join(d, NAME)


def read(d):
    """the hold of a run directory as a dict, {} for a marker that is not readable JSON (it still holds), None when there is none"""
    p = path(d)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        return h if isinstance(h, dict) else {}
    except (OSError, ValueError):
        return {}


def write(d, who, why, words="", keep=True):
    """put the marker on a run directory (made when it does not exist); keep=True: an existing marker stays as it is (the first word counts). -> the marker"""
    os.makedirs(d, exist_ok=True)
    old = read(d)
    if old is not None and keep:
        return old
    h = dict(id=os.path.basename(os.path.normpath(d)), who=who, when=dt.datetime.now().isoformat(timespec="seconds"), why=why, words=words)
    P.atomic_write(path(d), json.dumps(h, indent=1, ensure_ascii=False) + "\n")
    return h


def clear(d, words):
    """remove the marker (kept as HOLD.cleared.<date>.<n> with the words that cleared it); -> the marker that was there. Exits when there is none or no words"""
    if not words or not words.strip():
        sys.exit("hold.py clear needs --words: the person's own words that allow this run to start (a chat's judgement does not clear a hold)")
    h = read(d)
    if h is None:
        sys.exit(f"{os.path.basename(d)}: no hold")
    n = 0
    while os.path.exists(f"{path(d)}.cleared.{dt.date.today().isoformat()}.{n}"):
        n += 1
    done = dict(h, cleared=dt.datetime.now().isoformat(timespec="seconds"), cleared_words=words.strip())
    P.atomic_write(f"{path(d)}.cleared.{dt.date.today().isoformat()}.{n}", json.dumps(done, indent=1, ensure_ascii=False) + "\n")
    os.remove(path(d))
    return h


def note_in_ledger(run_id, text):
    """append `text` to the notes of the ledger row of the run (under the lock of the ledger); False when the run has no row"""
    import ledger as L
    with P.locked(P.path("ledger")):
        rows = L.load()
        row = next((r for r in rows if r["id"] == run_id), None)
        if row is None:
            return False
        row["notes"] = (row.get("notes") or "").rstrip("; ") + ("; " if row.get("notes") else "") + text
        L.save(rows)
    L.render()
    return True


def holds(records):
    """[(run id, marker)] of every run directory with a hold"""
    out = []
    for name in sorted(os.listdir(records)) if os.path.isdir(records) else []:
        h = read(os.path.join(records, name))
        if h is not None:
            out.append((name, h))
    return out


def line(run_id, h):
    return f"{run_id}: since {h.get('when', '?')} by {h.get('who', '?')}: {h.get('why', '?')}; the person's words: {h.get('words') or '(none recorded)'}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("set")
    s.add_argument("id")
    s.add_argument("--by", required=True, help="the chat or tool that sets it")
    s.add_argument("--reason", required=True)
    s.add_argument("--words", default="", help="the person's words (or 'relayed by <chat>: ...' for a second-hand instruction)")
    c = sub.add_parser("clear")
    c.add_argument("id")
    c.add_argument("--words", default="")
    sh = sub.add_parser("show")
    sh.add_argument("id", nargs="?")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.cmd:
        ap.error("a command (set, clear, show) or --selftest")
    records = P.path("run_records")
    if a.cmd == "set":
        h = write(os.path.join(records, a.id), a.by, a.reason, a.words)
        print("hold: " + line(a.id, h) + ("" if h.get("who") == a.by and h.get("why") == a.reason else "   (a hold was there already: the first one stays)"))
        return 0
    if a.cmd == "clear":
        d = os.path.join(records, a.id)
        h = clear(d, a.words)
        text = f"hold cleared {dt.date.today().isoformat()} on the person's words: {a.words.strip()!r} (hold was: {h.get('why', '?')}, set by {h.get('who', '?')} {h.get('when', '?')})"
        print(f"{a.id}: hold cleared; the run may be started again")
        if not note_in_ledger(a.id, text):
            print(f"note: {a.id} has no ledger row, the words are only in {os.path.relpath(path(d), P.root())}.cleared.*")
        return 0
    todo = [(a.id, read(os.path.join(records, a.id)))] if a.id else holds(records)
    todo = [(i, h) for i, h in todo if h is not None]
    print("\n".join(line(i, h) for i, h in todo) if todo else "no holds")
    return 0


def selftest():
    T = tempfile.mkdtemp(prefix="ar_hold_")
    old = os.environ.get("AR_PROJECT")
    try:
        os.makedirs(f"{T}/runs")
        json.dump(dict(ledger="LEDGER.csv", ledger_md="LEDGER.md", run_records="runs", tag_prefix="t"), open(f"{T}/project.json", "w"))
        os.environ["AR_PROJECT"] = T
        d = os.path.join(T, "runs", "R1")
        assert read(d) is None and holds(f"{T}/runs") == []
        h = write(d, "chat A", "exit 75", "stop it")
        assert h["who"] == "chat A" and read(d)["words"] == "stop it" and os.path.isdir(d)
        assert write(d, "chat B", "other", "x")["who"] == "chat A", "the first hold stays"
        assert write(d, "chat B", "other", "x", keep=False)["who"] == "chat B"
        os.makedirs(os.path.join(T, "runs", "R2"))
        open(path(os.path.join(T, "runs", "R2")), "w").write("not json")
        assert read(os.path.join(T, "runs", "R2")) == {}, "an unreadable marker still holds"
        assert [i for i, _ in holds(f"{T}/runs")] == ["R1", "R2"]
        for bad in ("", "   "):
            try:
                clear(d, bad)
                raise AssertionError("clear without the person's words must be refused")
            except SystemExit as e:
                assert "needs --words" in str(e)
        try:
            clear(os.path.join(T, "runs", "R3"), "go")
            raise AssertionError("clear of a run without a hold must say so")
        except SystemExit as e:
            assert "no hold" in str(e)
        # the ledger: a note is appended to the row, a run without a row gets none
        with open(f"{T}/LEDGER.csv", "w", newline="", encoding="utf-8") as f:
            f.write("id,status,notes\nR1,stopped,old note\n")
        h = clear(d, "Mirko: resume it")
        assert read(d) is None and h["who"] == "chat B" and len([f for f in os.listdir(d) if f.startswith("HOLD.cleared.")]) == 1
        assert note_in_ledger("R1", "hold cleared on the person's words") is True and note_in_ledger("R9", "x") is False
        notes = open(f"{T}/LEDGER.csv", encoding="utf-8").read()
        assert "old note; hold cleared on the person's words" in notes, notes
        write(d, "chat A", "again")                      # a second hold and a second clear: the records of the first stay
        clear(d, "Mirko: ok")
        assert len([f for f in os.listdir(d) if f.startswith("HOLD.cleared.")]) == 2
        # the command line
        assert main(["set", "R5", "--by", "chat C", "--reason", "waiting for the person", "--words", "relayed by chat C: stay stopped"]) == 0 and read(os.path.join(T, "runs", "R5"))
        assert main(["show"]) == 0 and main(["show", "R5"]) == 0
        assert main(["clear", "R5", "--words", "Mirko: go"]) == 0 and read(os.path.join(T, "runs", "R5")) is None
        print("hold selftest passed")
    finally:
        if old is None:
            os.environ.pop("AR_PROJECT", None)
        else:
            os.environ["AR_PROJECT"] = old
        shutil.rmtree(T, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
