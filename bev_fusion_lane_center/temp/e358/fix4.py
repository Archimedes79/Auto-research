import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/training/train/jwire_train.py", [
("""    s.add_argument("--pair-run", default=None, help="the pair run of the VAL records (default e312_p_s<seed>)")
    s.add_argument("--folds", type=int, default=5)""", """    s.add_argument("--pair-run", default=None, help="the pair run of the VAL records (default e312_p_s<seed>)")
    s.add_argument("--ends", default="kp", choices=JD.ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (needs --pair-run of an attach model)")
    s.add_argument("--folds", type=int, default=5)"""),
("""    r.add_argument("--name", default=None, help="the run folder (default e325_w_s<seed>; mandatory with --pair-run)")""", """    r.add_argument("--ends", default="kp", choices=JD.ENDS, help="the pair ends of the records: the keypoints (default) or the ATTACHED ends of the pair run (E358: needs --pair-run of an attach model)")
    r.add_argument("--name", default=None, help="the run folder (default e325_w_s<seed>; mandatory with --pair-run)")"""),
("""        run_pipeline(a.seed, a.name, a.heads, a.epochs, a.workers, a.max_train, pair_run=a.pair_run)""", """        run_pipeline(a.seed, a.name, a.heads, a.epochs, a.workers, a.max_train, pair_run=a.pair_run, ends=a.ends)"""),
("""    recs = JD.load_records(JD.cache_path("val", a.seed, pair_run=a.pair_run), a.seed, a.pair_run)
    res = cv(recs, a.folds, a.epochs, seed=0)
    print(f"seed {a.seed}, pair run {a.pair_run or JD.default_pair_run(a.seed)}: scene-wise""", """    recs = JD.load_records(JD.cache_path("val", a.seed, pair_run=a.pair_run, ends=a.ends), a.seed, a.pair_run, a.ends)
    res = cv(recs, a.folds, a.epochs, seed=0)
    print(f"seed {a.seed}, pair run {a.pair_run or JD.default_pair_run(a.seed)}, {a.ends} ends: scene-wise"""),
])
print("ok")
