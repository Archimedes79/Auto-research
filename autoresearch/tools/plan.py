"""The status of the nodes of PLAN.md, read from the schedule and the ledger (so the plan keeps the reasoning and cannot go stale).

    python autoresearch/tools/plan.py status [--plan path/to/PLAN.md]    # one line per node: schedule items by state, ledger rows, last verdict

A node is a row of the plan's table: its first cell is the node id (N1), its second cell names the experiment (starts with E240 or t10.dual).
Without a PLAN.md the schedule is the list of experiments: every distinct `node` of schedule.csv is one node and names the experiment (t22, E265), so the plan file is optional.
Schedule items belong to a node by their `node` column, ledger rows by the experiment id (the row id is E240 or continues with . or _: E240.a, t10.dual_s0), the verdict is the
`decide.py <date>: KEEP|CUT|TIE|INCONCLUSIVE` token of the newest row that has one (a verdict word in prose is ignored); the state of an item comes
from the run records (schedule.derived). The status column of PLAN.md, where there is one, is only a note of the planner.
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402

VERDICT = re.compile(r"decide\.py [\d-]+: (KEEP|CUT|TIE|INCONCLUSIVE)")  # only the token decide.py writes; a verdict word in prose is not a verdict


def nodes(text):
    """[(node id, experiment id or '', title)] of the table rows of a PLAN.md"""
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 2 or not re.fullmatch(r"[A-Za-z]\d+[a-z]?", cells[0]):
            continue
        m = re.match(r"([A-Za-z]+\d+[a-z]?(?:\.[A-Za-z0-9_]+)?)\b", cells[1].replace("`", ""))
        out.append((cells[0], m.group(1) if m else "", cells[1]))
    return out


def nodes_of_schedule(sched_rows):
    """[(node id, experiment id, title)] without a plan file: one node per distinct `node` of the schedule, in order of first appearance (the node names the experiment: t22, E265)"""
    seen, out = set(), []
    for r in sched_rows:
        n = (r.get("node") or "").strip()
        if n and n not in seen:
            seen.add(n)
            out.append((n, n, n))
    return out


def status(plan_text, sched_rows, ledger_rows):
    """list of dicts: node, experiment, items {state: n}, ledger {status: n}, verdict; the nodes come from the plan's table, or from the schedule when there is no plan (plan_text None)"""
    out = []
    for node, exp, title in (nodes(plan_text) if plan_text is not None else nodes_of_schedule(sched_rows)):
        items = [r for r in sched_rows if (r.get("node") or "").strip() == node]
        rows = [r for r in ledger_rows if exp and re.fullmatch(re.escape(exp) + r"(?:[._].*)?", r["id"])]  # E265 takes E265.a, not E2650; t10.dual takes t10.dual_s0
        st, ls = {}, {}
        for r in items:
            st[r["state"]] = st.get(r["state"], 0) + 1
        for r in rows:
            ls[r.get("status") or "planned"] = ls.get(r.get("status") or "planned", 0) + 1
        v = [w for r in sorted(rows, key=lambda r: r.get("started") or "") for w in VERDICT.findall(r.get("decision") or "")[-1:]]  # the last verdict word of each row, oldest row first
        out.append(dict(node=node, experiment=exp, title=title, items=st, ledger=ls, verdict=v[-1] if v else ""))
    return out


def line(s):
    items = ", ".join(f"{k} {n}" for k, n in sorted(s["items"].items())) or "-"
    ledger = ", ".join(f"{k} {n}" for k, n in sorted(s["ledger"].items())) or "-"
    return f"{s['node']:5} {s['experiment'] or '-':12} schedule: {items:28} ledger: {ledger:24} verdict: {s['verdict'] or '-'}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["status"])
    ap.add_argument("--plan", default=None)
    a = ap.parse_args()
    import ledger as L
    import schedule as S
    where = [os.path.join(os.path.dirname(P.path("schedule")), "PLAN.md"), os.path.join(P.root(), "plan", "plan.md"), os.path.join(P.root(), "PLAN.md")]
    plan = a.plan or next((w for w in where if os.path.isfile(w)), "")
    if a.plan and not os.path.isfile(plan):
        sys.exit(f"no plan at {plan}")
    text = open(plan, encoding="utf-8").read() if plan else None  # no plan file: the nodes are the nodes of the schedule (the list of experiments)
    rows = status(text, S.derived(S.load()), L.load())
    if not rows and text is None:
        sys.exit(f"no plan (looked in {', '.join(where)}) and no schedule nodes (pass --plan)")
    for s in rows:
        print(line(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
