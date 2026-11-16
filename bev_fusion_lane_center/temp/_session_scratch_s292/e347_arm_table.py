"""E347: print an arm's gate numbers in the layout of the control table (runs/e347_<arm>_s<S>/jmaps_val.json, key gate) next to the control of the same seed.
usage: python3 temp/_session_scratch_s292/e347_arm_table.py g0 o0 ...    (arm letter + seed)"""
import json
import sys


def rows(d):
    g = d["gate"]
    out = {}
    for key, name in ((("ours", "at_best"), "own rule, best"), (("ours", "at_thr0_3"), "own rule, 0.3"), (("ours_alt", "at_best"), "other, best"), (("ours_alt", "at_thr0_3"), "other, 0.3")):
        r = g[key[0]][key[1]]
        out[name] = (r["thr"]["ent"], r["thr"]["ext"], r["f1_tol3"], r["f1_tol1_5"], r["crowded_recall"]["ent"], r["crowded_recall"]["ext"], r["dup_share"], r["edge"]["ent"] or r["edge"]["ext"])
    return out


for a in sys.argv[1:]:
    arm, S = a[0], int(a[1:])
    d = json.load(open(f"runs/e347_{arm}_s{S}/jmaps_val.json"))
    c = json.load(open(f"runs/e332_ma_s{S}/jmaps_val_legacy.json"))
    print(f"== arm {arm} seed {S} (frames {d['frames']}); zone IoU {d.get('zone_iou')}, core IoU {d.get('core_iou')} (control {c.get('zone_iou')}, {c.get('core_iou')})")
    ra, rc = rows(d), rows(c)
    print(f"{'read-out':16s} {'thr e/x':10s} {'F1@3':>7s} {'F1@1.5':>7s} {'crowdE':>7s} {'crowdX':>7s} {'dup':>7s} edge | control own-rule best: F1@3 F1@1.5 crowdE")
    for k, v in ra.items():
        print(f"{k:16s} {v[0]:.2f}/{v[1]:.2f} {v[2]:7.4f} {v[3]:7.4f} {v[4]:7.4f} {v[5]:7.4f} {v[6]:7.4f} {'Y' if v[7] else 'n'}")
    cb = rc["own rule, best"]
    print(f"  control own rule best: F1@3 {cb[2]:.4f}  F1@1.5 {cb[3]:.4f}  crowdE {cb[4]:.4f}  dup {cb[6]:.4f}")
    # the stage-1 clauses at the arm's rule read-out (g: own rule = peak + threshold; o: own rule = threshold alone, i.e. the 'other' row in the scorer's naming when the run's rule is the threshold alone)
    rule_key = "own rule, best"
    v = ra[rule_key]
    print(f"  deltas of the rule read-out against the control: F1@3 {v[2] - cb[2]:+.4f}  F1@1.5 {v[3] - cb[3]:+.4f}  crowded entry recall {v[4] - cb[4]:+.4f}")
    gs = d.get("gate_sweep", {}).get("ours", {})
    for kind in ("ent", "ext"):
        for r in gs.get(kind, {}).get("rows", []):
            if abs(r["thr"] - 0.1) < 1e-6:
                print(f"  0.10 row ({kind}): n_pred {r['n_pred']} n_dup {r['n_dup']} dup share {r['n_dup'] / max(r['n_pred'], 1):.4f}")
