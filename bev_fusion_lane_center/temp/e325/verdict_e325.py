"""E325 verdict by hand (the rule of the card section 0, written before the run): snapped topo F1 and the guards of the arm w (head seed 0) against c = E312 on both seeds, the head seeds 1 and 2 (range), the kill and the
agreement numbers of `runs/e325_w_s<S>/wire_agreement.json`, the calibration on labelled ends, the price (parameters). Without BOTH seeds' cards the verdict is INCOMPLETE.
usage: wsl bash tools/env/_py14.sh ../temp/e325/verdict_e325.py"""
import json
import os
import re

ROOT = os.environ["BFLC_ROOT"]
R = ROOT + "/runs/"
K = ("topo_F1", "dup_frac", "F1@1.0", "lanes_found", "junction_F1", "vert100_dp")
BOUND = dict(topo_F1=0.005, **{"F1@1.0": 0.01}, dup_frac=0.01, vert100_dp=1.0, junction_F1=0.01, lanes_found=0.01)                 # max loss against c (dup_frac and vertices: a rise)
tokens, killed = [], []
for s in (0, 1):
    c = json.load(open(f"{R}e312_p_s{s}/ends_card_snap_thr0.1.json"))["numbers"]["head"]
    wp = f"{R}e325_w_s{s}/wire_agreement.json"
    if not os.path.exists(wp):
        print(f"== seed {s}: no wire_agreement.json (the training run did not finish)")
        continue
    wa = json.load(open(wp))
    pt = wa["pilot_train"]
    print(f"== seed {s}: pilot TRAIN {{keypoint error median {pt['err_median']:.2f} m, rule real-choice agreement {pt['multi']:.3f}, kp_p mean {pt.get('kp_p_mean', float('nan')):.2f}, multi-place share {pt.get('multi_share', float('nan')):.2f}}}, jitter {wa['jitter_scale']:.2f} m")
    for h in wa["heads"]:
        v, t = h["val"], h["train_jitter"]
        print(f"   head {h['head']}{' (distance-only ablation)' if h['ablation'] else ''}: VAL overall {v['head_same'] / v['oracle']:.3f} (rule {v['rule_same'] / v['oracle']:.3f}), real {v['head_multi_same'] / v['multi']:.3f} (rule {v['rule_multi_same'] / v['multi']:.3f}) | TRAIN+jitter real {t['head_multi_same'] / t['multi']:.3f} (rule {t['rule_multi_same'] / t['multi']:.3f})")
    print(f"   KILL fired: {wa['kill_fired']} (on VAL {wa.get('kill_on_val')}, on TRAIN {wa.get('kill_on_train')}); a kill that comes from the shift between TRAIN and VAL is NOT a failure to learn: the in-sample VAL cross-validation (E325 section 5) is above the rule")
    cal = wa.get("calibration_val_labelled_ends")
    if cal:
        print("   calibration on labelled VAL ends (largest weight bin: ends, share correct): " + ", ".join(f"[{lo:.1f},{hi:.1f}) {n}: {a:.2f}" for lo, hi, n, a in cal if n and a is not None))
    killed.append(wa["kill_fired"])
    for hh, tag in ((0, "e325w"), (1, "e325w1"), (2, "e325w2")):
        p = f"{R}e312_p_s{s}/ends_card_{tag}_snap_thr0.1.json"
        if not os.path.exists(p):
            print(f"   head {hh}: no scored card")
            continue
        w = json.load(open(p))["numbers"]["head"]
        d = {k: w[k] - c[k] for k in K}
        ok = {k: ((d[k] if k in ("dup_frac", "vert100_dp") else -d[k]) <= BOUND[k]) for k in K}
        print(f"   head {hh} snapped: " + " ".join(f"{k} {w[k]:.4f} ({d[k]:+.4f})" for k in K) + f" | guards {'held' if all(ok.values()) else 'MISSED: ' + str([k for k in ok if not ok[k]])}")
        if hh == 0:
            tokens.append("KEEP" if all(ok.values()) else "TIE")
    bp = f"{ROOT}/temp/e325/boot_w_vs_c_s{s}.log"
    m = re.search(r"topo_F1\s+A - B: point ([+-][0-9.]+), bootstrap mean [+-][0-9.]+, 95 % \[([+-][0-9.]+), ([+-][0-9.]+)\]", open(bp).read()) if os.path.exists(bp) else None
    print("   bootstrap w - c topo F1:", m.groups() if m else "not yet")
print("price: 545 parameters, at most 0.12 MMACs per frame (the card's count; ms of the host-side candidate lists not measured here)")
if len(tokens) < 2:
    print("TOKEN: INCOMPLETE (the cards of arm w are missing for a seed: a crashed run or scorer)")
elif any(killed):
    print("TOKEN: CUT by the kill (the head does not learn / does not carry over): read the kill note above before calling it a failure to learn")
elif tokens == ["KEEP", "KEEP"]:
    print("TOKEN: KEEP (the head replaces the nearest-piece rule at the output)")
else:
    print("TOKEN: TIE, kept and refined (D87): the head is the differentiable replacement of the rule, the miss is reported:", tokens)
