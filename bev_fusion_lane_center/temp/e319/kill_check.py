"""E319 gate after stage 1 (card section 0, revised 2026-11-14): the pair stage (stage 2, about 2.8 slot hours for two seeds) runs on a seed only when the maps with the exchange GAIN over E307 of the same seed:
keypoint F1 (thr 0.3, 3 m, distance only, mean of entry and exit) OR the entry recall beyond 35 m at least +0.01 above E307; it does not run when the keypoint F1 is more than 0.03 below E307 (the kill of the card) or when neither number gains.
Exit codes: 0 = gate met, stage 2 runs; 1 = gate not met (no gain, or the kill): stage 2 does not run on this seed, a result and not a failure; 2 = the check itself failed (missing or unreadable jmaps_val.json, any exception):
chain.sh does not run stage 2 and reports the chain as failed. The last line printed is `GATE met` / `GATE not met` / `GATE check failed`, which chain.sh reads together with the exit code (a Python traceback also exits 1).
usage: kill_check.py <seed> [<maps run with the block> [<reference maps run>]]   (defaults e319_m_s<seed>, e307_n_s<seed>; the optional runs are for tests)"""
import json
import os
import sys
import traceback

GAIN, KILL = 0.01, 0.03
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def nums(run):
    d = json.load(open(os.path.join(ROOT, "runs", run, "jmaps_val.json")))
    k = d["keypoints"]
    return dict(kp=sum(k[f"kp_ours_tol3.0_noang_{t}"]["f1"] for t in ("ent", "ext")) / 2, far=d["range_city_split"]["range_ours_ent_gt35"]["recall"])


def main(argv):
    S = argv[1]
    run, ref = (argv[2] if len(argv) > 2 else f"e319_m_s{S}"), (argv[3] if len(argv) > 3 else f"e307_n_s{S}")
    a, b = nums(run), nums(ref)
    dkp, dfar = a["kp"] - b["kp"], a["far"] - b["far"]
    print(f"seed {S}: {run} against {ref}: keypoint F1 {a['kp']:.4f} vs {b['kp']:.4f} ({dkp:+.4f}), entry recall beyond 35 m {a['far']:.4f} vs {b['far']:.4f} ({dfar:+.4f}); gain bar +{GAIN:.2f} on either, kill below -{KILL:.2f} keypoint F1")
    if dkp < -KILL:
        print("GATE not met: the maps are worse than the reference by more than the kill margin (the kill of the card)")
        return 1
    if dkp >= GAIN - 1e-9 or dfar >= GAIN - 1e-9:
        print("GATE met: a maps gain, stage 2 runs on this seed")
        return 0
    print("GATE not met: no maps gain (keypoint F1 and far-range entry recall both below +%.2f): stage 2 is not run on this seed" % GAIN)
    return 1


if __name__ == "__main__":
    try:
        rc = main(sys.argv)
    except Exception:  # noqa: BLE001  (a crash must not look like a verdict: exit 2)
        traceback.print_exc()
        print("GATE check failed")
        rc = 2
    sys.exit(rc)
