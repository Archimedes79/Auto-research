"""scratch: the dry check of jgraph on REAL files: the stored dump of e312_p_s0 / e324_s_s0 written as if it were the two arms' dumps into a temp runs dir; then `commands` logic; and the format of the smoke dump (bf16, 2 frames) against the stored dump."""
import os
import pickle
import sys
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
R = os.environ["BFLC_ROOT"]
sys.path.insert(0, os.path.join(R, "evaluation"))
import jgraph as JG  # noqa: E402

smoke = os.path.join(R, "temp", "_session_scratch_e351", "smoke_e312", "e351_g1_e312_p_s0", "ends_val.pkl")
st = pickle.load(open(os.path.join(R, "runs", "e312_p_s0", "ends_val.pkl"), "rb"))
sm = pickle.load(open(smoke, "rb"))
for k in st[0]:
    a, b = st[0][k], sm[0][k]
    same = type(a) == type(b) and (not hasattr(a, "dtype") or (a.dtype == b.dtype and a.shape == b.shape))
    print(f"{k:8s} stored {type(a).__name__} {getattr(a, 'dtype', '')} {getattr(a, 'shape', '')} | smoke {type(b).__name__} {getattr(b, 'dtype', '')} {getattr(b, 'shape', '')} {'OK' if same else 'DIFFERENT'}")
assert set(st[0]) == set(sm[0]), (set(st[0]) ^ set(sm[0]))
print("rows", [f["row"] for f in st[:3]], [f["row"] for f in sm[:3]])
JG.ROOT = R
with tempfile.TemporaryDirectory() as td:
    for run in ("e312_p_s0", "e324_s_s0", "e312_p_s1", "e324_s_s1"):
        src = os.path.join(R, "runs", run)
        frames = pickle.load(open(os.path.join(src, "ends_val.pkl"), "rb"))
        seed, attach = JG.run_info(src)
        for arm in JG.ARMS:
            JG.write_dump(src, JG.dump_dir(run, arm, td), frames, dict(arm=arm))
        for line in JG.score_lines(run, seed, attach):
            print(line)
        for arm in JG.ARMS:
            for fl in JG.score_flags(attach):
                bad, name = JG.check_score_ready(JG.dump_dir(run, arm, td), seed, fl, JG.PREFIX, src=src)
                print(f"  [{run} {arm}] {' '.join(fl)}: {'ready, would write ' + name if not bad else 'NOT READY ' + str(bad)}")
