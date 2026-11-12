"""The adoption rule of the simplification arms E331 - E335 (D84, D85): a removal is adopted when the delivered graph loses at most the noise band against the control of the same seed, on BOTH seeds and on BOTH read-outs.

    wsl bash tools/env/_py14.sh jrule_removal.py --arm e331_e_s --control e312_p_s [--seeds 0 1] [--max-loss 0.01]      exit 0 = ADOPT, 1 = not adopted, 2 = a card is missing
    wsl bash tools/env/_py14.sh jrule_removal.py selftest

Read-outs (D85: the unsnapped one is the differentiable one and counts; the snapped one, the rule step of E311, is reported and must hold too): `plain` = `ends_card_thr0.1.json` (connectors at p >= 0.1 drawn from the keypoints) and
`snap` = `ends_card_snap_thr0.1.json` (the same, ends snapped to the delivered pieces), both written by `jends_eval.py score` into `runs/<run>/` (`--thr 0.1` with and without `--snap`); `attach` = `ends_card_attach_thr0.1.json` (E320: connector ends attached by the network, no snap) once E320 is the reference. The arm run of seed S is `<arm><S>`, the control `<control><S>`.
Loss of an arm against its control (head numbers of the cards): topo F1 at most `--max-loss` (0.01, the noise band of S), F1@1.0 at most 0.01, junction F1 at most 0.03, dup_frac at most +0.01, vertices per 100 m at most +1.0 (the guards of the cards E315 - E320).
Written before the first run of the arms; the verdict of a card is this exit code together with the numbers printed (by hand: replay, as the cards say).
"""
import argparse
import json
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
READOUTS = {"plain": ("unsnapped (differentiable)", "ends_card_thr0.1.json"), "snap": ("snapped (rule step)", "ends_card_snap_thr0.1.json"), "attach": ("attached (E320, differentiable)", "ends_card_attach_thr0.1.json")}
GUARDS = (("F1@1.0", -0.01), ("junction_F1", -0.03), ("dup_frac", None), ("vert100_dp", None))


def head(run, name):
    p = os.path.join(ROOT, "runs", run, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))["numbers"]["head"]


def check(arm_h, ctl_h, max_loss):
    """-> (list of (metric, difference, ok)), all ok"""
    rows = []
    d = arm_h["topo_F1"] - ctl_h["topo_F1"]
    rows.append(("topo_F1", d, d >= -max_loss))
    for m, lim in GUARDS:
        d = arm_h[m] - ctl_h[m]
        ok = d >= lim if lim is not None else (d <= 0.01 if m == "dup_frac" else d <= 1.0)
        rows.append((m, d, ok))
    return rows, all(r[2] for r in rows)


def run(arm, control, seeds, max_loss, readouts=("plain", "snap")):
    verdict, missing = True, False
    for label, name in (READOUTS[r] for r in readouts):
        for s in seeds:
            a, c = head(f"{arm}{s}", name), head(f"{control}{s}", name)
            if a is None or c is None:
                print(f"  {label}, seed {s}: card missing ({arm}{s} or {control}{s}: {name})")
                missing = True
                continue
            rows, ok = check(a, c, max_loss)
            verdict &= ok
            print(f"  {label}, seed {s}: topo F1 {a['topo_F1']:.4f} against {c['topo_F1']:.4f} | " + ", ".join(f"{m} {d:+.4f}{'' if o else ' FAIL'}" for m, d, o in rows) + f" -> {'ok' if ok else 'NOT ok'}")
    if missing:
        print("verdict: INCOMPLETE (a card is missing)")
        return 2
    print("verdict:", "ADOPT the removal (within the band on both seeds and both read-outs)" if verdict else "NOT adopted")
    return 0 if verdict else 1


def selftest():
    base = dict(topo_F1=0.68, **{"F1@1.0": 0.87, "junction_F1": 0.32, "dup_frac": 0.17, "vert100_dp": 19.0})
    rows, ok = check(dict(base, topo_F1=0.672), base, 0.01)
    assert ok and abs(rows[0][1] + 0.008) < 1e-9
    assert not check(dict(base, topo_F1=0.668), base, 0.01)[1]                                         # -0.012 on topo F1
    assert not check(dict(base, dup_frac=0.19), base, 0.01)[1] and not check(dict(base, vert100_dp=20.5), base, 0.01)[1]
    assert not check(dict(base, **{"F1@1.0": 0.855}), base, 0.01)[1] and check(dict(base, **{"F1@1.0": 0.862}), base, 0.01)[1]
    assert not check(dict(base, junction_F1=0.285), base, 0.01)[1] and check(dict(base, junction_F1=0.30), base, 0.01)[1]
    print("jrule_removal selftest ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="run-name prefix of the arm, the seed is appended (e.g. e331_e_s)")
    ap.add_argument("--control", default="e312_p_s", help="run-name prefix of the control (e312_p_s; e320_a_s once E320 is the reference)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max-loss", type=float, default=0.01)
    ap.add_argument("--readouts", nargs="+", default=["plain", "snap"], choices=sorted(READOUTS), help="plain = unsnapped, snap = snapped; attach = the E320 attached read-out (re-based arms: --readouts attach)")
    a = ap.parse_args()
    sys.exit(run(a.arm, a.control, a.seeds, a.max_loss, a.readouts))
