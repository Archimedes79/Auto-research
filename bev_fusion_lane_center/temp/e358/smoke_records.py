"""E358 scratch: real-data smoke of the end kind on the stored dumps (CPU, read only, no file written): the first N VAL frames of `runs/e324_s_s0` and `runs/e312_p_s0`.

    wsl bash temp/e358/run_scratch.sh temp/e358/smoke_records.py [N]

1. kp records of the NEW `jwire_data.val_records` equal, field by field, the records of the ORIGINAL file (copy of the tree file before E358: temp/_session_scratch_e358/orig_jwire_data.py), on both pair runs: kp bit for bit.
2. att records of e324_s_s0: the pair ends are the `att_pos` / `att_dir` of the dump, `gts` equal the kp records' (the match stays on the keypoints), the pilot numbers (error of the matched ends, share within 0.5 m).
3. att on E312's pair run is refused before anything is read.
"""
import os
import pickle
import sys
import types

import numpy as np

ROOT = os.environ["BFLC_ROOT"]
import jwire_data as NEW  # noqa: E402
import jwiring as JWI  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
old = types.ModuleType("jwire_data_orig")
old.__file__ = os.path.join(ROOT, "evaluation", "jwire_data.py")
exec(compile(open(os.path.join(ROOT, "temp", "_session_scratch_e358", "orig_jwire_data.py"), encoding="utf-8").read(), old.__file__, "exec"), old.__dict__)

memo = {}
real_load = JWI._load


def load_once(a):
    key = (a.seed, a.run)
    if key not in memo:
        memo[key] = real_load(a)
    return memo[key]


JWI._load = load_once
FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")


def same(a, b):
    return a["row"] == b["row"] and a["scene"] == b["scene"] and all(np.array_equal(a[k], b[k], equal_nan=True) and a[k].dtype == b[k].dtype for k in FIELDS)


for run, seed in (("e312_p_s0", 0), ("e324_s_s0", 0)):
    r_old = old.val_records(seed, run, 0.1, N)
    r_new = NEW.val_records(seed, run, 0.1, N)
    assert len(r_old) == len(r_new) == N and all(same(a, b) for a, b in zip(r_old, r_new)), f"{run}: kp records differ from the original code"
    print(f"{run}: {N} kp records identical to the original code (pairs {sum(len(r['pairs']) for r in r_new)})")
r_kp = r_new
r_att = NEW.val_records(0, "e324_s_s0", 0.1, N, ends="att")
dump = {f["row"]: f for f in pickle.load(open(os.path.join(ROOT, "runs", "e324_s_s0", "ends_val.pkl"), "rb"))}
n_pairs = n_differ = 0
for rk, ra in zip(r_kp, r_att):
    e = dump[ra["row"]]
    k = e["kp_pos"].shape[0] // 2
    p = 1.0 / (1.0 + np.exp(-e["logit"])) * e["allowed"]
    ia, ib = np.nonzero(p >= 0.1)
    want = np.array([np.r_[e["att_pos"][i], e["att_dir"][i], e["att_pos"][k + j], e["att_dir"][k + j]] for i, j in zip(ia, ib)], float).reshape(-1, 8)
    assert np.array_equal(ra["pairs"], want), f"row {ra['row']}: the pair ends are not the attached ends of the dump"
    assert np.array_equal(ra["gts"], rk["gts"], equal_nan=True) and np.array_equal(ra["kp_p"], rk["kp_p"]), "the GT match / keypoint probabilities must not depend on the end kind"
    n_pairs += len(want)
    n_differ += int(not np.array_equal(ra["pairs"], rk["pairs"]))
assert n_pairs > 0 and n_differ > 0, "attached ends equal the keypoints everywhere: the test frames show nothing"
print(f"e324_s_s0 att: {N} records, {n_pairs} pairs; the pair ends are the dump's att_pos / att_dir, {n_differ} frames differ from the kp records, gts and kp_p equal")
for name, recs in (("kp ", r_kp), ("att", r_att)):
    ps = NEW.pilot_stats(recs)
    print(f"  {name}: matched ends {ps['matched_ends']}, error median {ps['err_median']:.3f} m, mean {ps['err_mean']:.3f}, p90 {ps['err_p90']:.3f}; near05 {ps['near05']:.3f}; multi-place share {ps['multi_share']:.3f}; rule agreement overall {ps['overall']:.3f}, real choices {ps['multi']:.3f}")
try:
    NEW.val_records(0, "e312_p_s0", 0.1, N, ends="att")
    raise SystemExit("att on E312's pair run was not refused")
except AssertionError as e:
    print("att on e312_p_s0 refused:", str(e)[:100])
print("smoke ok")
