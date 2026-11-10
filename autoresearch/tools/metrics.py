"""Loop metrics of the research system (ROADMAP P3): what a loop ran, decided and predicted, read from the ledger, the decision files of decide.py
and the TEST log. Standard library only.

    python3 autoresearch/tools/metrics.py                  # one row per loop
    python3 autoresearch/tools/metrics.py --loop 18 -v     # one loop and how each ledger row was counted
    python3 autoresearch/tools/metrics.py --md             # the markdown table (ledger.py render puts it at the head of LEDGER.md)
    python3 autoresearch/tools/metrics.py --selftest

A loop is a range of experiment numbers (project.json "loops": {"18": [209, 233], "19": [234, null]}); run E219.rc1 belongs to the loop of 219.
Columns
  runs / failed   ledger rows that ran / of them failed, timed out, stalled, blocked or refused (gpu_busy); a run stopped on request ("stopped", exit 75) is a run, no failure
  GPU h           sum of the durations in hours (before the tick of ROADMAP P2 a chained run's duration includes its wait: an upper bound)
  exps            distinct experiment numbers with a run
  decisions       decide.py verdicts (one per decision*.json file) and the written decisions of the other rows: KEEP / CUT / TIE /
                  INC(onclusive) / other; a written "rule met" counts as KEEP, "rule not met" as CUT, everything else written ("confirmed",
                  "fixed", ...) as other; "reference" rows are controls and not counted; a row whose decision carries "decide.py" is counted
                  through its decision file only (its token stands before or behind the prose; the prose is read without it)
  pred hit        written predictions met / judged ("prediction ... met"; any "missed" or "not met" after the word counts as missed): the
                  person or lead who wrote it knew the prediction's own tolerance, the decision files do not, so they give only the error
  mean err        mean absolute prediction error of the decision files, in the metric most of them are about (its name in brackets)
  TEST            rows of the TEST log that name the loop and used TEST
Written decisions are read with simple patterns; `-v` lists every row with its class, so a person can check the count.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import project as P  # noqa: E402
from decide import split_decision  # noqa: E402  (the decision cell: decide.py writes it and says how to split it)

FAILED = ("failed", "timeout", "stalled", "gpu_busy", "blocked")
VERDICTS = ("KEEP", "CUT", "TIE", "INCONCLUSIVE")


def number(eid):
    m = re.match(r"E(\d+)", eid or "")
    return int(m.group(1)) if m else None


def loop_of(n, loops):
    for name, (lo, hi) in loops.items():
        if n is not None and lo <= n and (hi is None or n <= hi):
            return name
    return None


def written(text):
    """(verdict class, prediction class) of a written decision"""
    t = (text or "").lower()
    v = "reference" if t.strip() in ("reference", "control") else None
    if v is None:
        m = re.search(r"\brule\b.{0,90}?\b(not met|met)\b", t)
        v = ("CUT" if m.group(1) == "not met" else "KEEP") if m else ("other" if t.strip() else None)
    p = None
    i = t.find("prediction")
    if i >= 0:
        seg = t[i:]
        p = "missed" if re.search(r"\bmissed\b|\bnot met\b", seg) else "met" if re.search(r"\bmet\b", seg) else None
    return v, p


def collect(root=None, cfg=None):
    root, cfg = root or P.root(), cfg or P.cfg()
    loops = {k: tuple(v) for k, v in cfg.get("loops", {}).items()}
    rows = []
    p = os.path.join(root, cfg["ledger"])
    if os.path.exists(p):
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    decisions = {}
    for d in [f for base in cfg.get("experiment_dirs", ["experiments"]) for f in glob.glob(os.path.join(root, base, "*", "decision*.json"))]:
        try:
            d = json.load(open(d, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict):
            decisions.setdefault(d["id"] if d.get("id") is not None else len(decisions), d)       # one per experiment id, also when a copy sits in two card folders (the open one first)
    decisions = list(decisions.values())
    test_log = os.path.join(root, cfg.get("test_log", ""))
    test_rows = open(test_log, encoding="utf-8").read().splitlines() if os.path.isfile(test_log) else []
    out = {name: dict(loop=name, runs=0, failed=0, gpu_s=0.0, exps=set(), KEEP=0, CUT=0, TIE=0, INCONCLUSIVE=0, other=0, keep2=0,
                      pred_met=0, pred_n=0, errs=[], errs_by={}, test=0, detail=[]) for name in loops}
    for d in decisions:
        L = out.get(loop_of(number(str(d.get("id"))), loops))
        if L is None:
            continue
        v = d.get("verdict", "")
        L[v if v in VERDICTS else "other"] += 1
        if v == "KEEP" and len(d.get("diffs") or []) >= 2:
            L["keep2"] += 1
        m = re.search(r"error ([-+]?\d+\.\d+)", str(d.get("prediction") or ""))
        if m:
            L["errs"].append(abs(float(m.group(1))))
            L["errs_by"].setdefault(str(d.get("metric") or ""), []).append(abs(float(m.group(1))))     # errors of different metrics are not one unit
        L["detail"].append(f"  {d.get('id')}.{d.get('variant')} vs {d.get('control')}: decision file {v} ({str(d.get('prediction') or '')[:70]})")
    for r in rows:
        n = number(r["id"])
        L = out.get(loop_of(n, loops))
        if L is None or r.get("status") in ("planned", "held", ""):             # held: refused by a hold while it waited, it never ran
            continue
        L["runs"] += 1
        L["failed"] += r.get("status") in FAILED
        try:
            L["gpu_s"] += float(r.get("duration_s") or 0)
        except ValueError:
            pass
        L["exps"].add(n)
        dec = r.get("decision") or ""
        v, pr = written(split_decision(dec)[0])        # the prose, whether the decide.py token stands before or behind it
        if "decide.py" in dec:
            v = None  # the verdict of this arm is counted through its decision file
        if v in VERDICTS + ("other",):
            L[v] += 1
        if pr:
            L["pred_n"] += 1
            L["pred_met"] += pr == "met"
        L["detail"].append(f"  {r['id']:12} {r.get('status', ''):8} {v or '-':9} pred {pr or '-':6} | {dec[:90]}")
    for line in test_rows:
        m = re.match(r"\|\s*[\d/-]+\s*\|\s*(\d+)[a-z]?\s*\|(.*)", line)
        if m and m.group(1) in out and "not used" not in m.group(2).lower():
            out[m.group(1)]["test"] += 1
    return out


def table(out, md=False):
    head = ["loop", "runs", "failed", "GPU h", "exps", "KEEP", "CUT", "TIE", "INC", "other", "KEEP x2", "pred hit", "mean err", "TEST"]
    lines = []
    for name, L in out.items():
        if not L["runs"] and not L["detail"]:
            continue
        hit = f"{L['pred_met']}/{L['pred_n']}" if L["pred_n"] else "-"
        err = "-"
        if L["errs_by"]:                                 # the metric most decisions of the loop are about; named when the loop has several
            m = max(L["errs_by"], key=lambda k: len(L["errs_by"][k]))
            err = f"{sum(L['errs_by'][m]) / len(L['errs_by'][m]):.4f}" + (f" ({m})" if m else "")
        lines.append([name, L["runs"], L["failed"], f"{L['gpu_s'] / 3600:.1f}", len(L["exps"]), L["KEEP"], L["CUT"], L["TIE"], L["INCONCLUSIVE"],
                      L["other"], L["keep2"], hit, err, L["test"]])
    if md:
        return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(str(x) for x in r) + " |" for r in lines])
    w = [max(len(str(x)) for x in col) for col in zip(head, *lines)] if lines else [len(h) for h in head]
    return "\n".join("  ".join(str(x).rjust(n) for x, n in zip(r, w)) for r in [head] + lines)


def selftest():
    import shutil
    import tempfile
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, "experiments", "E2_x"))
    with open(os.path.join(root, "LEDGER.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "status", "duration_s", "decision"])
        w.writeheader()
        w.writerows([dict(id="E1.a", status="ok", duration_s="3600", decision="rule (Q >= 0.7) not met; prediction Q 0.7 missed"),
                     dict(id="E1.c", status="ok", duration_s="1800", decision="reference"),
                     dict(id="E2.a", status="ok", duration_s="1800", decision="rule met | decide.py 2026-10-25: KEEP (effect +0.0200)"),
                     dict(id="E2.c", status="failed", duration_s="", decision=""),
                     dict(id="E3.a", status="planned", duration_s="", decision=""),
                     dict(id="E1.h", status="held", duration_s="0", decision=""),               # refused by a hold while it waited: not a run
                     dict(id="E9.a", status="ok", duration_s="7200", decision="confirmed; prediction met")])
    json.dump(dict(id="E2", verdict="KEEP", variant="a", control="c", diffs=[0.02, 0.018], band=0.01, metric="Q", prediction="prediction +0.015, measured +0.0200 (error +0.0050)"),
              open(os.path.join(root, "experiments", "E2_x", "decision.json"), "w"))
    for n, metric, err in ((3, "vert100", "+11.1503"), (4, "Q", "-0.0150")):       # another unit in the same loop must not enter the Q mean
        for d in (os.path.join(root, "results", "experiments" if n == 4 else "experiments", f"E{n}_y"),) + ((os.path.join(root, "experiments", "E4_y"),) if n == 4 else ()):
            os.makedirs(d)         # E4 sits in the second card folder, and a copy of it in the first: it counts once
            json.dump(dict(id=f"E{n}", verdict="TIE", variant="a", control="c", metric=metric, prediction=f"prediction x, measured y (error {err})"),
                      open(os.path.join(d, "decision.json"), "w"))
    open(os.path.join(root, "TEST.md"), "w").write("| 2026-10-25 | 1 | champion | E2 | S 0.7 | once |\n| 2026-10-26 | 2 | not used | - | - | - |\n")
    cfg = dict(P.DEFAULTS, ledger="LEDGER.csv", test_log="TEST.md", loops={"1": [1, 5], "2": [6, None]}, experiment_dirs=["experiments", "results/experiments"])
    out = collect(root, cfg)
    L = out["1"]
    assert (L["runs"], L["failed"], len(L["exps"])) == (4, 1, 2), L
    assert abs(L["gpu_s"] - 7200) < 1e-6
    assert (L["KEEP"], L["CUT"], L["other"], L["keep2"]) == (1, 1, 0, 1), L
    assert (L["pred_met"], L["pred_n"]) == (0, 1) and sorted(L["errs_by"]["Q"]) == [0.005, 0.015] and L["errs_by"]["vert100"] == [11.1503], L
    assert "0.0100 (Q)" in table(out), table(out)
    assert written("prediction lanes found >= 0.74 not met") == ("other", "missed")
    assert written("rule (Q >= 0.7) not met: prediction met") == ("CUT", "met")
    assert L["test"] == 1 and out["2"]["test"] == 0 and out["2"]["other"] == 1 and out["2"]["pred_met"] == 1
    assert "| 1 | 4 | 1 | 2.0 | 2 | 1 | 1 |" in table(out, md=True)
    shutil.rmtree(root)
    print("metrics selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loop")
    ap.add_argument("-v", action="store_true")
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    out = collect()
    if a.loop:
        if a.loop not in out:
            sys.exit(f"no loop {a.loop!r}; known: {', '.join(out)}")
        out = {a.loop: out[a.loop]}
    print(table(out, a.md))
    if a.v:
        for name, L in out.items():
            print(f"\nloop {name}:")
            print("\n".join(L["detail"]))


if __name__ == "__main__":
    main()
