"""The rule control for the junction pair head (E287; the literature scan of 2026-11-11, Nilsson 2025: a simple rule can match a learned pair head at signalised intersections): pairs from the entry and exit keypoints by geometry
and lane order alone, no learning, no image.

    wsl bash tools/env/_py14.sh jrule_control.py score --run e287_p2_s0 [--run e284_h2_s0 ...]      pair recall / precision of the rule and of the head on the SAME keypoints (the head at logit >= 0), VAL 873
    wsl bash tools/env/_py14.sh jrule_control.py selftest

The rule (written before the numbers): entries and exits are grouped (`jgt._groups`: heading within 25 degrees, 6 m along and across) and ranked (0 = leftmost). A group of entries A and a group of exits B are CONNECTED when the exit
group lies ahead of the entry group (the chord ahead of both headings, `qends.allowed` for the group centres) and the turn between the headings is a class: straight (|turn| < 35 degrees), left (35 .. 150), right (-150 .. -35);
for every class only the connected exit group with the shortest chord counts. Lane matching inside a connected pair (A, B), by normalised rank r = rank / (k - 1) (0.5 for k = 1): straight: every entry lane goes to the
exit lane of the nearest normalised rank; left: the entries of rank 0 (and of rank 1 when the arm has 4 or more lanes) go to the exit of the nearest normalised rank; right: the same for the last rank(s); an arm with one
lane connects to every class. The rule needs the keypoints to be there: a missing entry or exit is a missing pair, which is why it is a control for the pair decision and not for the whole head.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub in ("evaluation", "training", "training/grid", "training/data", "tools/viz", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import jgt as J  # noqa: E402

STRAIGHT, TURN_MAX = 35.0, 150.0


def _norm_rank(r, k):
    return 0.5 if k <= 1 else r / (k - 1)


def rule_pairs(ent, ext):
    """ent, ext: (n, 4) / (m, 4) valid keypoints x, y, tx, ty -> (n, m) bool matrix of the pairs the rule delivers"""
    out = np.zeros((len(ent), len(ext)), bool)
    if not len(ent) or not len(ext):
        return out
    ga, gb = J._groups(ent), J._groups(ext)
    ra, rb = J._ranks(ent, ga), J._ranks(ext, gb)
    for a in range(ga.max() + 1):
        ia = np.where(ga == a)[0]
        ca, ta = ent[ia, :2].mean(0), ent[ia, 2:].mean(0)
        ta /= np.linalg.norm(ta) + 1e-9
        best = {}
        for b in range(gb.max() + 1):
            ib = np.where(gb == b)[0]
            cb, tb = ext[ib, :2].mean(0), ext[ib, 2:].mean(0)
            tb /= np.linalg.norm(tb) + 1e-9
            ch = cb - ca
            L = np.linalg.norm(ch)
            if L < 2.0 or ch @ ta <= 0 or ch @ tb <= 0:
                continue
            turn = (np.degrees(np.arctan2(ta[0] * tb[1] - ta[1] * tb[0], ta @ tb)))
            cls = "straight" if abs(turn) < STRAIGHT else ("left" if STRAIGHT <= turn <= TURN_MAX else ("right" if -TURN_MAX <= turn <= -STRAIGHT else None))
            if cls and (cls not in best or L < best[cls][0]):
                best[cls] = (L, b)
        ka = len(ia)
        for cls, (_, b) in best.items():
            ib = np.where(gb == b)[0]
            kb = len(ib)
            if cls == "straight" or ka == 1:
                lanes = list(range(ka))
            elif cls == "left":
                lanes = [r for r in range(ka) if r == 0 or (ka >= 4 and r == 1)]
            else:
                lanes = [r for r in range(ka) if r == ka - 1 or (ka >= 4 and r == ka - 2)]
            for r in lanes:
                i = ia[np.where(ra[ia] == r)[0][0]]
                target = _norm_rank(r, ka)
                j = ib[np.argmin([abs(_norm_rank(rb[x], kb) - target) for x in ib])]
                out[i, j] = True
    return out


def _match(kp, kd, valid, G, pos_tol=4.0, ang=30.0):
    out = -np.ones(len(G), int)
    for g, e in enumerate(G):
        d = np.linalg.norm(kp - e[:2], axis=1)
        a = np.degrees(np.arccos(np.clip(kd @ e[2:], -1, 1)))
        ok = valid & (d <= pos_tol) & (a <= ang)
        if ok.any():
            out[g] = int(np.argmin(np.where(ok, d, 1e9)))
    return out


def score(a):
    import qgraph_view as QV
    res = {}
    for run in a.run:
        frames = {f["token"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", run, "ends_val.pkl"), "rb"))}
        tot = dict(gt=0, rule_hit=0, rule_n=0, rule_ok=0, head_hit=0, head_n=0, head_ok=0)
        for sid, (name, toks) in QV.val_scenes().items():
            for t in toks:
                if t not in frames:
                    continue
                jg = J.junction_gt(QV.load_gt(t))
                f = frames[t]
                K = f["logit"].shape[0]
                gt_pairs = set() if jg is None else {(int(p[0]), int(p[1])) for p in jg["pairs"]}
                tot["gt"] += len(gt_pairs)
                kp, kd = f["kp_pos"], f["kp_dir"]
                ve, vx = f["valid"][0], f["valid"][1]
                ie, ix = np.where(ve)[0], np.where(vx)[0]
                R = rule_pairs(np.concatenate([kp[:K][ie], kd[:K][ie]], 1), np.concatenate([kp[K:][ix], kd[K:][ix]], 1))
                Hd = (1 / (1 + np.exp(-f["logit"]))) * f["allowed"] >= 0.5
                if jg is not None:
                    ma = _match(kp[:K], kd[:K], ve, jg["ent"])
                    mb = _match(kp[K:], kd[K:], vx, jg["ext"])
                    inv_a = {int(k): g for g, k in enumerate(ma) if k >= 0}
                    inv_b = {int(k): g for g, k in enumerate(mb) if k >= 0}
                else:
                    inv_a = inv_b = {}
                for name_, M, rows, cols in (("rule", R, ie, ix), ("head", Hd, np.arange(K), np.arange(K))):
                    got = set()
                    for ii, jj in zip(*np.where(M)):
                        i, j = int(rows[ii]), int(cols[jj])
                        tot[f"{name_}_n"] += 1
                        key = (inv_a.get(i, -1), inv_b.get(j, -1))
                        if key in gt_pairs:
                            tot[f"{name_}_ok"] += 1
                            got.add(key)
                    tot[f"{name_}_hit"] += len(got)
        r = lambda x, y: x / max(y, 1)  # noqa: E731
        res[run] = dict(gt_pairs=tot["gt"], rule=dict(recall=r(tot["rule_hit"], tot["gt"]), precision=r(tot["rule_ok"], tot["rule_n"]), delivered=tot["rule_n"]),
                        head=dict(recall=r(tot["head_hit"], tot["gt"]), precision=r(tot["head_ok"], tot["head_n"]), delivered=tot["head_n"]))
        print(run, json.dumps(res[run]))
    os.makedirs(os.path.join(ROOT, "runs", "e287"), exist_ok=True)
    json.dump(res, open(os.path.join(ROOT, "runs", "e287", "rule_control.json"), "w"), indent=1)


def selftest():
    # two entries side by side going north, two exits going north (straight), one exit going west far to the left (left turn)
    ent = np.array([[-1.75, -8.0, 0.0, 1.0], [1.75, -8.0, 0.0, 1.0]])
    ext = np.array([[-1.75, 8.0, 0.0, 1.0], [1.75, 8.0, 0.0, 1.0], [-10.0, 0.0, -1.0, 0.0]])
    R = rule_pairs(ent, ext)
    assert R[0, 0] and R[1, 1] and not R[0, 1] and not R[1, 0], R                      # entry 1 -> exit 1, entry 2 -> exit 2 (D82)
    assert R[0, 2] and not R[1, 2], R                                                     # the left turn only from the leftmost lane
    assert not rule_pairs(ent[:0], ext).any() and not rule_pairs(ent, ext[:0]).any()
    print("jrule_control selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    s = sub.add_parser("score")
    s.add_argument("--run", nargs="+", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else score(a)
