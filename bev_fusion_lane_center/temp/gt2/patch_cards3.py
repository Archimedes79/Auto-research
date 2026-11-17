def edit(path, pairs):
    t = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert a in t, (path, a[:60])
        t = t.replace(a, b)
    open(path, "w", encoding="utf-8").write(t)


edit("experiments/E311_snap-connector-ends/experiment.md", [
    ("(`temp/jr/snapconn.py` `add_connectors_snap` (scored by `temp/gt2/snap_score.py: the scorer of record `qends_eval.cmd_score` with only the connector function replaced), pairs at p >= t", "x") if False else ("`temp/jr/snapconn.py` `add_connectors_snap` (scored by `temp/gt2/snap_score.py`: the scorer of record `qends_eval.cmd_score` with only the connector function replaced), pairs at p >= t",
     "`qends_eval.add_connectors(snap=True)` (promoted from the scratch `snapconn.py` after the critic pass; scored by `jends_eval.py score --snap --thr t`, the scorer of record with only the connector function changed), pairs at p >= t"),
    ("rule applied by `temp/gt2/e311_rule.py`", "rule applied by `jdiag.py snaprule e287_p2 e305_e`"),
    ("`temp/gt2/snap/<run>_thr<t>.json`", "`runs/<run>/ends_card_snap_thr<t>.json` (the first confirmation scores were written by a scratch scorer into `temp/gt2/snap/` with the same code path and copied there)"),
])
edit("experiments/E309_pair-delivery-threshold/experiment.md", [
    ("(`temp/gt2/thr_sweep.py <run> <seed> 0.3 0.2 0.1`: the scorer of record, only `--thr` changes, the run's own ends_card.json is left alone)", "(`jends_eval.py score --run <run> --seed <S> --thr t`: the scorer of record, only `--thr` changes; the card goes to `runs/<run>/ends_card_thr<t>.json`, the card of record is only written at p >= 0.5; the first sweep used a scratch wrapper `temp/gt2/thr_sweep.py`)"),
])
print("ok")
