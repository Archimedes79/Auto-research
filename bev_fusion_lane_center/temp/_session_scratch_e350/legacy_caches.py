"""read-only: E325's real cache files (no meta) still load for E312's pair run and are refused for another; their names are what cache_path gives"""
import os
import sys

R = os.environ["BFLC_ROOT"]
sys.path.insert(0, R + "/evaluation")
import jwire_data as JD  # noqa: E402

for kind, seed, nmax in (("val", 0, 0), ("val", 1, 0), ("train", 0, 0), ("train", 1, 0), ("train", 0, 200), ("val_clean", 0, 24)):
    p = JD.cache_path(kind, seed, nmax)
    assert os.path.exists(p), p
    if kind == "train" and nmax == 0 and seed == 1:
        print("exists (not loaded):", os.path.basename(p))
        continue
    recs = JD.load_records(p, seed)
    recs2 = JD.load_records(p, seed, f"e312_p_s{seed}")
    try:
        JD.load_records(p, seed, f"e324_s_s{seed}")
        raise SystemExit(f"{p}: accepted for another pair run")
    except AssertionError as e:
        why = str(e)[-60:]
    print(f"{os.path.basename(p)}: {len(recs)} records, loads for E312 ({len(recs2)}), refused for e324_s_s{seed} ({why})")
    try:
        JD.save_records(p, recs, seed)
        raise SystemExit("an existing real cache would have been overwritten")
    except AssertionError:
        pass
print("legacy caches ok (never written)")
