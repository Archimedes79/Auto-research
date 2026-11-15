"""one-off (E275): bring the card text in line with the code after the critic pass (names, ties, strict radius, reproduction gate, dump). The rule (section 0 verdict rule, bands, guards, predictions) is NOT changed."""
p = "experiments/E275_node-readout-suppression/experiment.md"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep("keep a node only if no kept node with a higher e lies within 1.5 m; the links are read between the kept nodes; scoring cards runs/e275/<run>_n/card.json",
    "keep a node only if no kept node with a higher e lies closer than 1.5 m (strictly: at exactly 1.5 m both stay); the links are read between the kept nodes; scoring cards runs/e275/<run>.close1.5_t0.5/card.json")
rep("or its output depends on the order of the input nodes, or the reproduction of step 2 misses the sweep numbers of E274 section 7 by more than 0.005: stop and search the code before any other number is read",
    "or its output depends on the order of the input nodes (exact ties of e are the one allowed exception: they are broken by the lower index and counted in the card), or the reproduction of step 2 misses the sweep numbers of E274 section 7 by more than 0.005 (enforced by `--repro` before any card is written): stop and search the code before any other number is read")
rep("and keep a node unless a kept node lies within R metres (Euclidean, BEV metres);",
    "and keep a node unless a kept node lies closer than R metres (Euclidean, BEV metres; strictly: at exactly R both stay); nodes with exactly equal e (a saturated sigmoid) go in index order, so only for exact ties does the kept set depend on the order of the input, and the card counts those pairs (`tied_pairs_within_radius`);")
rep("(`qgraph_run_eval.py score --readout close:1.5 --exp e275`)",
    "(`qgraph_run_eval.py score --exp e275 --readouts plain@0.5 close:1.5@0.5 close:1.5@0.4 plain@0.4 --curve 1.0 1.25 1.75 2.0 --dump --repro temp/e274/sweep_<run>.json`; `--exp` is required with `--readouts`; one card per read-out in `runs/e275/<run>.<spec>/`, `card['run']` is the model, `card['readout']` the step's counts and ms per frame)")
rep("are written to `temp/e275/pred_<run>.pt` (scratch): the analysis of step 3 and the radius curve run on CPU from these files, so the GPU is needed once per model.",
    "are written to `temp/e275/pred_<run>.pkl` (scratch; nodes with e >= 0.2: position, direction, flags, cell, section, e, with the values as the scorer sees them; no links, no assignment): the analysis of step 3 (dropped nodes against the lane GT) runs on CPU from these files; the radius curve (graph numbers only, e >= 0.5) is computed with `--curve` during the same GPU pass, so the GPU is needed once per model.")
open(p, "w", encoding="utf-8").write(s)
print("card patched")
