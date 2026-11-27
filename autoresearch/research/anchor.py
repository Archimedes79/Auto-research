"""The anchor test of a network: evaluate a finished run again with the code of the working tree, into a fresh folder (nothing stored is
reused), and compare every number with the run's result.json. Run with the problem's interpreter from the problem folder:

    python -m research.anchor <problem> <run id> [--tolerance 0]

Exit 0 when every number agrees (bit for bit at tolerance 0), 1 otherwise, 2 when the test itself failed; the last line of stdout is always
one json answer (with "error" on exit 2), so a caller never has to read a traceback."""
from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import traceback

from research.paths import read_json, record_dir, run_dir


def main(problem: str, rid: str, tolerance: float = 0.0) -> int:
    rec = read_json(os.path.join(record_dir(problem, rid), "result.json"))
    tmp = tempfile.mkdtemp(prefix=f"anchor_{rid}_")
    os.environ["RESEARCH_EVAL_OUT"] = tmp                       # Evaluation.defaults.out: a fresh place, so nothing stored is reused
    try:
        from Evaluation import Evaluation
        p = read_json(os.path.join(os.getcwd(), "problem.json"))
        ev = Evaluation(project=os.getcwd())
        done = ev.run(run_dir(problem, rid), rec["evaluation"].get("frames", rec["split"]), metrics=tuple(rec["evaluation"]["metrics"]), device=p["evaluate"].get("device", "cuda"),
                      workers=0, progress=0)
        now = ev.show_numbers(done["id"])["numbers"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    then = rec["numbers"]
    def differs(a, b):
        a, b = float(a), float(b)
        if math.isnan(a) or math.isnan(b):
            return not (math.isnan(a) and math.isnan(b))           # an undefined number agrees only with an undefined one
        return abs(a - b) > tolerance

    diff = {k: (then.get(k), now.get(k)) for k in sorted(set(then) | set(now)) if k not in then or k not in now or differs(then[k], now[k])}
    print(json.dumps({"run": rid, "agree": not diff, "differ": diff, "tolerance": tolerance}, default=str))
    return 0 if not diff else 1


if __name__ == "__main__":
    args = sys.argv[1:]
    try:
        tol = float(args[args.index("--tolerance") + 1]) if "--tolerance" in args else 0.0
        sys.exit(main(args[0], args[1], tol))
    except Exception as err:                                           # noqa: BLE001 - every outcome is a json answer
        traceback.print_exc()
        print(json.dumps({"run": args[1] if len(args) > 1 else "", "agree": False, "error": f"{type(err).__name__}: {err}"[:1500]}))
        sys.exit(2)
