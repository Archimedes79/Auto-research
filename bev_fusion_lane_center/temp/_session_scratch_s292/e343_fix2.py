p = "experiments/E343_test-e324s/experiment.md"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
rep('labelled "without interval".', 'labelled "without interval"; the D79 guards apply unchanged in the fallback (a miss is "contradicted, without interval").')
rep("The run records the sha256 of the imported files (`jends_*`, `qends_eval`, `jmaneuver`, `qgraph_*`, `jtest_eval`, `Evaluation/`) before and after, no commit or edit of these files from the final critic pass until the run ends is agreed with the lane lead, and the post-run VAL re-check of E321's habit stays.",
    "The code records its own sha256 map (`code_sha256`: the imported lane code files, `jtest_eval`, `qends_eval`, `jends_eval`, `jmaneuver`, `qgraph_*`, `Evaluation/stats.py`) in every card and every dump's `_head.json` and compares it at the end of `all`: a change is printed as CODE DRIFT and written into `results.md` (non-fatal, as in E321, where the launcher noted it); every dump also records the device and the checkpoint's sha256 and is NOT reused when code, device or checkpoint differ (a stale VAL dump cannot be reused by a rehearsal after a code change; E321's own dumps carry no provenance and are read-only inputs validated by their hash). No commit or edit of the imported files from the final critic pass until the run ends is agreed with the lane lead, and the post-run VAL re-check of E321's habit stays. The card, `results.md` and the log name the device.")
rep("If the GPU branch is outside the tolerance too, the check has failed (kill).", "If the GPU branch is outside the tolerance too, the check has failed (kill). **Expect the GPU branch:** E321 measured 3.0e-3 on vertices per 100 m against the 3e-3 bound, and this rehearsal has four read-outs per seed, so a miss on that one number is not unlikely; then the GPU branch applies as written. **The rehearsal uses the flags of the TEST line pointing at VAL files** (B's fresh VAL dump from `dump-b`, `--control-card` = E312's VAL card in the E321 layout, `--control-ends-dump` = E312's stored VAL pair dump) so that the TEST layout of the control card, the flagged B dump and the 1e-9 branch are exercised; the exact command is in the module docstring and in the result of this section.")
a = "(`--control-ends-dump` is used: R1 is confirmed;"
assert s.count(a) == 1
s = s.replace(a, "**`--no-stack` is the ONLY permitted change of this line:** it drops arm k when k's own check (criterion 1 or 2) fails on VAL; it is decided from VAL only, written into this card and into the log row BEFORE the launch and never after any TEST number; the rehearsal is then re-run with `--no-stack` to show that w and p pass without k (without the flag a failure of k makes `verify` and the VAL `all` exit non-zero for everything). " + a)
open(p, "w", encoding="utf-8").write(s)
print("ok")
