import io

p = "experiments/E347_ends-on-the-2m-node-grid/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (a[:60], s.count(a))
    s = s.replace(a, b)


# finding 1: the pre-registered recipe text of arm g names the real offset supervision
rep("offset L1 at the cell of each GT end, zone / core rastered", "offset L1 (metres) on every cell within 3.0 m of the nearest end of the kind (`jgt.OFF_NEAR`, mask `joffm`; target = that end minus the cell centre; weight `OFF_W` 1.0; changed from 'the cell of each GT end' on 2026-11-15, see section 2b), zone / core rastered")
# the critic line
rep("a fresh `ar-critic` (mode code) delta pass on round 2: result entered here: (pending).",
    "a fresh `ar-critic` (mode code) delta pass on round 2 (2026-11-15 11:35): **ACCEPT**, nothing blocks the first GPU start; its findings, taken here: (1) the recipe text of arm g (done above); (2) **the duplicate clause is read at each run's best 3 m F1 threshold, but the pair stage consumes p >= `PEAK_MIN` = 0.1 (`jends.py`, top-K 32, no peak step for o): the duplicate share at the 0.10 row of `gate_sweep` (`n_dup` / `n_pred`) is quoted next to the clause in the result; the rule stays as written**; (3) **the kill after 500 steps can fire wrongly on arm o** (a focal head with a one-hot target is calibrated lower than a heat with positive weight 20: its peaks may sit below 0.3 at step 500): a zero count at 0.3 is checked against `gate_sweep` of `jmaps_val_ckpt.json` (n_pred down to 0.05), a head that has predictions at lower thresholds is not dead and the case is labelled 'by hand', not read as structural; (4) **grid parity: the control is quoted from the rescored `jmaps_val_legacy.json` (`gate.ours.at_best`, edge false) on the same 0.05 - 0.95 grid as the arms; the old best-threshold values 0.7561 / 0.7501 (grid 0.2 - 0.7) are at most equal to the new-grid best and would favour the arms**; the rescore of the control's threshold-alone read-out fires thousands of 1 m keypoints per frame at 0.05 (the keypoint loop is per point): tens of minutes per seed on the CPU; (5) operational: no `_pyr.sh`-style retry on a CUDA unknown error and `--retries 0` (an arm dies once; a restart resumes from `ckpt.pt`), arms `--workers 4` against 3 of the control.")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
