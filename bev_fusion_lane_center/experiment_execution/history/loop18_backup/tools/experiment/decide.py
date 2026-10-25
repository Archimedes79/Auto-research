"""Decision rule as code: keep / cut / tie / inconclusive from per-seed scores of a variant and its control.

    python decide.py --variant 0.746,0.741 --control 0.725,0.722 [--band 0.010] [--lower-better] [--cost-variant 31 --cost-control 628] [--id E123.D7 --write]

Rule (CLAUDE.md, habit 4):
  * effect = mean(variant) - mean(control); with equal seed counts the seeds are paired (seed i against seed i)
  * |effect| <= band                      -> TIE: take the cheaper variant (needs --cost-*; otherwise 'take the simpler one')
  * effect beyond the band, every paired seed on the same side -> KEEP (better) or CUT (worse)
  * effect beyond the band but a paired seed disagrees, or only one seed -> INCONCLUSIVE: add a seed, do not adopt
  * the band defaults to 0.010 (S); use the measured spread for other metrics (route success: 0.05 - 0.09)
  * --write puts the verdict into the decision column of the ledger row --id (tools/experiment/ledger.py)
This is the pre-registered decision, not a significance test: scene-level intervals come from evaluation/sparse_lanes/bootstrap.py.
"""
import argparse
import os
import subprocess
import sys


def decide(variant, control, band=0.010, lower_better=False, cost_variant=None, cost_control=None):
    sgn = -1.0 if lower_better else 1.0
    mv, mc = sum(variant) / len(variant), sum(control) / len(control)
    effect = sgn * (mv - mc)
    if len(variant) == len(control):
        diffs = [sgn * (v - c) for v, c in zip(variant, control)]
    else:
        diffs = [effect]
    if abs(effect) <= band:
        if cost_variant is not None and cost_control is not None:
            cheaper = "variant" if cost_variant < cost_control else "control"
            return "TIE", effect, f"inside the noise band {band}: take the cheaper one ({cheaper}: {cost_variant} vs {cost_control})"
        return "TIE", effect, f"inside the noise band {band}: take the simpler / cheaper one"
    same = all((d > 0) == (effect > 0) and abs(d) > 0 for d in diffs)
    if len(diffs) < 2 or not same:
        return "INCONCLUSIVE", effect, "beyond the band but " + ("only one seed" if len(diffs) < 2 else "a seed disagrees") + ": add a seed, do not adopt yet"
    return ("KEEP" if effect > 0 else "CUT"), effect, f"beyond the band {band} on all {len(diffs)} paired seeds"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="comma list, one score per seed")
    ap.add_argument("--control", required=True)
    ap.add_argument("--band", type=float, default=0.010)
    ap.add_argument("--lower-better", action="store_true")
    ap.add_argument("--cost-variant", type=float)
    ap.add_argument("--cost-control", type=float)
    ap.add_argument("--id", default="")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    v = [float(x) for x in a.variant.split(",")]
    c = [float(x) for x in a.control.split(",")]
    verdict, effect, why = decide(v, c, a.band, a.lower_better, a.cost_variant, a.cost_control)
    line = f"{verdict}: effect {effect:+.4f} ({why})"
    print(line)
    if a.write and a.id:
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "ledger.py"), "note", a.id, "--decision", line], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
