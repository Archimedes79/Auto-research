import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
apply(R + "training/train/jwire_train.py", [
    ("""    val_stub, live_stub = _record_stubs(seen)
    old_root, old_stubs, old_fns""", """    val_stub, live_stub = _record_stubs(seen)
    up = np.array([0.0, 1.0])
    near_rec = lambda r: dict(r, pairs=np.array([np.r_[[0.0, -0.6], up, [5.0, 20.0], up]]))                  # noqa: E731  (the entry 0.4 m from the end of the first piece: the unperturbed share is 1 and a jitter can move it)
    train_recs = [near_rec(JD.toy_record(i, i % 22)) for i in range(30)]
    old_root, old_stubs, old_fns"""),
    ("""JD.val_records, JD.live_records = val_stub, live_stub
        cache = lambda f: os.path.join(d, "temp", "jwire", f)  # noqa: E731
        att_run""", """JD.val_records, JD.live_records = val_stub, lambda *a, **k: [near_rec(r) for r in live_stub(*a, **k)]
        cache = lambda f: os.path.join(d, "temp", "jwire", f)  # noqa: E731
        att_run"""),
    ("""            unjit = real_tensors([JD.toy_record(i, i % 22) for i in range(30)])""", """            unjit = real_tensors(train_recs)"""),
    ("""            jit = [real_tensors([JD.toy_record(i, i % 22) for i in range(30)], JD.make_jitter(wa["jitter_scale"], d)) for d in range(3)]""", """            jit = [real_tensors(train_recs, JD.make_jitter(wa["jitter_scale"], d)) for d in range(3)]"""),
])
