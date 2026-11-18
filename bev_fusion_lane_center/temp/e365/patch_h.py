import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a
    s = s.replace(a, b)
rep("the prediction written from the TOKENS only, no GT field, by `lane_ptok_data.to_pred`, kept = p_none < 0.5, decodes rec / argmax;", "the prediction written from the TOKENS only by `lane_ptok_data.to_pred`, kept = p_none < 0.5, decodes rec / argmax; the real no-GT guarantee is that the head reads only `INPUTS` (`training/grid/test_jtoken.py` poison test), the writer asserts on every frame that the nodes are the kept tokens' own, e = 1 - p_none and the links are the dump's;")
rep("paired scene bootstrap against plain B; cards `runs/R/token_card_<decode>[_train][_drop..].json`, never overwritten), `ctl`", "paired scene bootstrap against plain B and against B' with the close-node step, g1; `--max N` for a partial read-out into `temp/e365/out/`; cards `runs/R/token_card_<decode>[_train][_drop..].json`, never overwritten), `ctl`")
rep("-> `runs/eval/e365_ctl/`), `check-layer1` (this scorer on B's stored dump = the E324 s base numbers to 1e-6, PASS / FAIL), `check-parity` (head from cache tokens vs tokens recomputed live from B'), `timing` (parameters, GMACs, ms of the head / B' / the whole path; GPU only when idle)", "-> `runs/eval/e365_ctl/<arm>_<ident>_s<S>_K<K>.json`), `check-layer1` (this scorer on B's stored dump = the E324 s base numbers to 1e-6, PASS / FAIL, prints the dump's sha256 and mtime; only a PASS stores plain B's records, which `score` needs), `check-parity` (head from cache tokens vs tokens recomputed live from B', matched by cell, unmatched valid tokens <= 0.5 %; `token_parity_<UTC>.json`), `timing` (parameters, GMACs, ms of the head / B' / the whole path, K from the cache of the run; GPU only when idle, this run.py job excluded; `token_timing_<UTC>.json` with `quiet` and `usable_for_gcost`)")
open(p, "w", encoding="utf-8").write(s)
