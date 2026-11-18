import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
# the helper, before _test_pair_run
("""def _test_pair_run():
    \"\"\"E350 plumbing""", """def _record_stubs(seen):
    \"\"\"-> (val_stub, live_stub): stand-ins for `JD.val_records` / `JD.live_records` that note their arguments (seed, pair run, end kind ...) in `seen` and return toy records\"\"\"
    def val_stub(seed, run, thr=0.1, nmax=0, ends="kp"):
        seen["val"].append((seed, run, ends))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):
        seen["live"].append((seed, split, nmax, pair_run, ends))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]
    return val_stub, live_stub


def _test_pair_run():
    \"\"\"E350 plumbing"""),
("""    seen = dict(val=[], live=[])

    def val_stub(seed, run, thr=0.1, nmax=0, ends="kp"):
        seen["val"].append((seed, run, ends))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):
        seen["live"].append((seed, split, nmax, pair_run, ends))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]

    def refused(f, *a, **k):""", """    seen = dict(val=[], live=[])
    val_stub, live_stub = _record_stubs(seen)

    def refused(f, *a, **k):"""),
("""    seen = dict(val=[], live=[])

    def val_stub(seed, run, thr=0.1, nmax=0, ends="kp"):
        seen["val"].append((seed, run, ends))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):
        seen["live"].append((seed, split, nmax, pair_run, ends))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]
    old_root, old_stubs, old_fns""", """    seen = dict(val=[], live=[])
    val_stub, live_stub = _record_stubs(seen)
    old_root, old_stubs, old_fns"""),
])
print("ok")
