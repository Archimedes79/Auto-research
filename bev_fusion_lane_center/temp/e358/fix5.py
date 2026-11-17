import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
("""    def val_stub(seed, run, thr=0.1, nmax=0):
        seen["val"].append((seed, run))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None):
        seen["live"].append((seed, split, nmax, pair_run))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]""", """    def val_stub(seed, run, thr=0.1, nmax=0, ends="kp"):
        seen["val"].append((seed, run, ends))
        return [JD.toy_record(i, i % 22) for i in range(873)]

    def live_stub(seed, split="train", thr=0.1, nmax=0, workers=3, bs=4, device=None, pair_run=None, ends="kp"):
        seen["live"].append((seed, split, nmax, pair_run, ends))
        return [JD.toy_record(i, i % 22) for i in range(nmax)]"""),
("""            assert seen == dict(val=[(0, "e324_s_s0")], live=[(0, "train", 30, "e324_s_s0")]), seen              # the pair run arrives at val_records AND at live_records""",
 """            assert seen == dict(val=[(0, "e324_s_s0", "kp")], live=[(0, "train", 30, "e324_s_s0", "kp")]), seen   # the pair run arrives at val_records AND at live_records"""),
("""            assert ck["args"]["pair_run"] == "e324_s_s0" and ck["args"]["seed"] == 0 and wa["pair_run"] == "e324_s_s0" and wa["seed"] == 0, (ck["args"], wa["pair_run"])""",
 """            assert ck["args"]["pair_run"] == "e324_s_s0" and ck["args"]["seed"] == 0 and wa["pair_run"] == "e324_s_s0" and wa["seed"] == 0, (ck["args"], wa["pair_run"])
            assert ck["args"]["ends"] == wa["ends"] == "kp" and os.path.exists(os.path.join(d, "runs", "e350_w_s0", "model_wn.pt")) and "no_jitter_head" in wa, "the end kind and the no-jitter head of every run"
            assert "kp ends" in open(os.path.join(d, "runs", "e350_w_s0", "log.txt")).read() and "no-jitter head (wn)" in open(os.path.join(d, "runs", "e350_w_s0", "log.txt")).read()"""),
("""            assert seen["val"][-1] == (0, "e312_p_s0") and seen["live"][-1] == (0, "train", 30, "e312_p_s0"), seen""",
 """            assert seen["val"][-1] == (0, "e312_p_s0", "kp") and seen["live"][-1] == (0, "train", 30, "e312_p_s0", "kp"), seen"""),
("""    _test_pair_run()
    print(f"jwire_train selftest ok""", """    _test_pair_run()
    _test_ends()
    print(f"jwire_train selftest ok"""),
])
print("ok")
