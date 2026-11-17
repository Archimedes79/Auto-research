import importlib.util, json, re, sys
R = "/mnt/c/Auto-research/bev_fusion_lane_center"
spec = importlib.util.spec_from_file_location("ar_testgate", "/mnt/c/Auto-research/autoresearch/tools/testgate.py")
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
cfg = json.load(open(f"{R}/project.json"))
grants = g.grant_ids(R, cfg)
print("decisions that grant a TEST check:", sorted(grants))
log = open(f"{R}/evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
row = ("| 2026-11-12/13 | 23 | the final lane system E312 (E307 maps + pair head `--e2e --scales`; connectors at p >= 0.1 snapped = the card of record of E312, and the same checkpoints not snapped = the differentiable read-out) "
       "against the plain delivered graph of B (E273 + close:1.5@0.5), seeds 0 and 1, TEST 833 frames | run id E321.t (`evaluation/jtest_eval.py all --split test --test-id E321.t`, card `experiments/E321_test-e312/experiment.md`) | - "
       "| Mirko's OK, once (D84, in chat 2026-11-12 23:00: \"Teste E312\"); read-outs, thresholds and metrics fixed on VAL, report only |\n")
for rid in ("E321.t", "E321.t_s0"):
    print(rid, "refusal:", repr(g.test_log_refusal(log + row, rid, grants)))
print("D84 granting words found by GRANT_STMT:", bool(g.GRANT_STMT.search(open(f"{R}/experiment_execution/decisions/D84.md", encoding="utf-8").read())))
