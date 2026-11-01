"""One-off: correct the commit / tag of a run whose snapshot failed (commit git-error), after the tag was set by hand.

    wsl python3 temp/loop19/fix_commit.py E240.a2 750d29f0d1 lcr/E240.a2 "snapshot race of five launchers started in the same second (read-tree BUG tempfile.c); tagged by hand at the commit that holds the code of the start"

Edits run.json and the ledger row under the locks of the tools (same API as fix_e234_records.py).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "autoresearch", "tools"))
import ledger as L  # noqa: E402
import project as P  # noqa: E402

rid, commit, tag, note = sys.argv[1:5]
p = os.path.join(os.path.dirname(P.path("ledger")), "runs", rid, "run.json")
with P.locked(p):
    r = json.load(open(p, encoding="utf-8"))
    r.update(commit=commit, tag=tag, git_note=note)
    P.atomic_write(p, json.dumps(r, indent=1) + "\n")
with P.locked(P.path("ledger")):
    rows = L.load()
    for x in rows:
        if x["id"] == rid:
            x["commit"] = commit
            x["notes"] = (x.get("notes", "") + "; " if x.get("notes") else "") + note
    L.save(rows)
L.render()
