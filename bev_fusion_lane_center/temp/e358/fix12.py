import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_check.py", [
("""    wsl bash tools/env/_py14.sh jwire_check.py twin --run e324_s_s0 --seed 0 --tag twin [--thr 0.1]            # CPU: the card of `jends_eval.py score --run <run> --seed <seed> --thr 0.1 --snap --wire twin --tag <tag>` against the
                                                                                                                 # stored card of the same run (`ends_card_snap_thr<thr>.json`)
    wsl bash tools/env/_py14.sh jwire_check.py live --seed 0 --pair-run e324_s_s0 [--max 24] [--workers 2] [--device cpu]   # the live pass (B and the pair run, `jwire_data.live_records`) on the first 24 `val_clean` frames against the
                                                                                                                 # records of the stored dump (`jwire_data.py val --seed <seed> --pair-run <run>` first); on a GPU (the default when there is
                                                                                                                 # one) it waits for room like `jends_eval.py dump` (`gpu_gate`); `--device cpu` needs neither (24 frames, a few minutes)
    wsl bash tools/env/_py14.sh jwire_check.py selftest
""", """    wsl bash tools/env/_py14.sh jwire_check.py twin --run e324_s_s0 --seed 0 --tag twin [--thr 0.1] [--read-out snap|primary]   # CPU: the card of `jends_eval.py score --run <run> --seed <seed> --thr 0.1 --snap --wire twin --tag <tag>` against the
                                                                                                                 # stored card of the same run (`ends_card_snap_thr<thr>.json`); `--read-out primary` (E358, with `--attach --snap-nogeom` in the
                                                                                                                 # scoring): the cards `ends_card_[<tag>_]snap_nogeom_attach_thr<thr>.json`
    wsl bash tools/env/_py14.sh jwire_check.py live --seed 0 --pair-run e324_s_s0 [--ends kp|att] [--max 24] [--workers 2] [--device cpu]   # the live pass (B and the pair run, `jwire_data.live_records`) on the first 24 `val_clean` frames against the
                                                                                                                 # records of the stored dump (`jwire_data.py val --seed <seed> --pair-run <run> [--ends att]` first); on a GPU (the default when
                                                                                                                 # there is one) it waits for room like `jends_eval.py dump` (`gpu_gate`); `--device cpu` needs neither (24 frames, a few minutes)
    wsl bash tools/env/_py14.sh jwire_check.py selftest
`--read-out` and `--ends` are one choice (snap = kp, primary = att): either fixes the other on both commands, both given must agree, none = snap / kp (E325 / E350).
"""),
("""  twin: EVERY field of the twin card equals the stored card (all keys of the stored card except `tag`, `wire`, `wire_run`, which differ by design; a key only the twin card has, `full` / `spec` of the scorer since E292, was
        not recorded in E312's stored cards and is not compared; the twin card must say `wire` = twin and the tag asked for, the stored card must be the rule's, both of the run and seed asked for): numbers, curves, wired""", """  twin: EVERY field of the twin card equals the stored card (all keys of the stored card except `tag`, `wire`, `wire_run`, which differ by design; a key only the twin card has, `full` / `spec` of the scorer since E292, was
        not recorded in E312's stored cards and is not compared; the twin card must say `wire` = twin and the tag asked for, the stored card must be the rule's, both of the run and seed asked for and of the read-out asked for:
        `attach` and `snap_nogeom` (in `snap_options`) true under primary, false under snap): numbers, curves, wired"""),
("""FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")                                                    # the arrays of a record (`jwire_data.record_of`)""", """FIELDS = ("ends", "attrs", "pairs", "kp_p", "gts")                                                    # the arrays of a record (`jwire_data.record_of`)
READ_OUT_ENDS = dict(snap="kp", primary="att")                                                        # the read-out of a stored card and the end kind of the records of its pair run (as `jwire_verdict.READ_OUT_ENDS`)"""),
("""def same(x, y):""", """def read_out_ends(read_out=None, ends=None):
    \"\"\"-> (read_out, ends) from the two options of the command line: either one fixes the other, both given must agree, none = (snap, kp)\"\"\"
    read_out = read_out or {v: k for k, v in READ_OUT_ENDS.items()}.get(ends, "snap")
    assert ends in (None, READ_OUT_ENDS[read_out]), f"--read-out {read_out} and --ends {ends} name different things (snap = kp, primary = att)"
    return read_out, READ_OUT_ENDS[read_out]


def same(x, y):"""),
("""def twin_check(run, seed, tag, thr=0.1):
    \"\"\"-> (identical, [what differs]) for the twin card `<tag>` of a run against its stored snapped card\"\"\"
    import qends_eval as QE
    names = [QE.card_name(NS(snap=True, thr=thr, tag=t)) for t in (None, tag)]""", """def twin_check(run, seed, tag, thr=0.1, read_out="snap"):
    \"\"\"-> (identical, [what differs]) for the twin card `<tag>` of a run against its stored card of the read-out (snap: `ends_card_snap_thr<thr>.json`; primary: `ends_card_snap_nogeom_attach_thr<thr>.json`)\"\"\"
    import qends_eval as QE
    primary = read_out == "primary"
    names = [QE.card_name(NS(snap=True, thr=thr, tag=t, attach=primary, snap_nogeom=primary)) for t in (None, tag)]"""),
("""        if c["seed"] != seed:
            diff.append(f"the {what} card is of seed {c['seed']}, not {seed}")""", """        if c["seed"] != seed:
            diff.append(f"the {what} card is of seed {c['seed']}, not {seed}")
        if bool(c.get("attach")) != primary or ("snap_nogeom" in (c.get("snap_options") or [])) != primary:
            diff.append(f"the {what} card is not the {read_out} read-out (attach {c.get('attach')}, snap_options {c.get('snap_options')})")"""),
("""def live_check(seed, pair_run, nmax=24, workers=2, device=None):
    import jwire_data as JD
    path = JD.cache_path("val", seed, pair_run=pair_run)
    assert os.path.exists(path), f"{path} is missing: `jwire_data.py val --seed {seed} --pair-run {pair_run}` builds the stored-dump records first"
    ref = {r["row"]: r for r in JD.load_records(path, seed, pair_run)}""", """def live_check(seed, pair_run, nmax=24, workers=2, device=None, ends="kp"):
    import jwire_data as JD
    path = JD.cache_path("val", seed, pair_run=pair_run, ends=ends)
    assert os.path.exists(path), f"{path} is missing: `jwire_data.py val --seed {seed} --pair-run {pair_run} --ends {ends}` builds the stored-dump records first"
    ref = {r["row"]: r for r in JD.load_records(path, seed, pair_run, ends)}"""),
("""    live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, device=dev, pair_run=pair_run)""", """    live = JD.live_records(seed, "val_clean", 0.1, nmax, workers, device=dev, pair_run=pair_run, ends=ends)"""),
])
print("ok")
