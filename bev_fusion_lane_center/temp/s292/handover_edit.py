s = open("HANDOVER.md", encoding="utf-8").read()
a = s.index("3. **E292** card v3"); b = s.index("4. **E299** time card v2")
new = '''3. **E292** card v3 (critic REVISE answered; no second pass unless the design changes). CODE NOT STARTED yet (nothing applied): plan = qgraph.py (`QGraph(mnms=(R, sigma))` + `node_stage`, `PHASE_*[9]`, `qgraph_loss` phase sets, selftest cases), qgates.gate phase 9 numbers-only, qgraph_train.py (`--soft-nms-mnms R SIGMA`, STEPS/LR[9]), run_config.py FLAGS, qpredict.load_model, scorer plumbing (`--full --base --spec --tag`, refuse-to-overwrite, pyramid assert; add_connectors / make_pairs unchanged), start scripts for c2 (FIRST) and a, READY. Reference outputs of B for the flag-off identity are saved: `bash tools/env/_py14.sh ../temp/s292/ref_b.py check` (`temp/s292/ref_b.pt`). jmaps_train.py imports qgraph.py (E332 / E333 maps stages): develop in `temp/_session_scratch_s292/` (the lookup skips that path), install with an atomic `mv` after the tests, never while a maps stage starts. Fresh critic pass on the code before READY.
'''
s = s[:a] + new + s[b:]
open("HANDOVER.md", "w", encoding="utf-8").write(s); print("ok")
