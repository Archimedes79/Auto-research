import sys
sys.path.insert(0, "autoresearch/tools")
import testgate as TG
log = open("bev_fusion_lane_center/evaluation/TEST_USAGE_LOG.md", encoding="utf-8").read()
row = ('| 2026-11-15 | 23 | the finished design E324 s (E307 maps + pair head `--e2e --scales --maps-w 1.0 --steps 6000 --geom-w 0.5 --attach`) read out snapped, PRIMARY (attached ends + snap-nogeom wiring) and snap-nogeom, and the stack E338 (E324 s heads on the delivered graph of c2, snapped), against E312 snapped (the stored TEST numbers of E321) and the plain delivered graph of B, seeds 0 and 1, TEST 833 frames '
       '| run id E343.t (`evaluation/jtest_eval.py all --split test --test-id E343.t --run-prefix e324_s_s`, card `experiments/E343_test-e324s/experiment.md`) | - '
       '| Mirko\'s OK, once (D95, 2026-11-14, "2. ja"; the three readings confirmed 2026-11-14 about 22:25 with "mach alles"): E324 s with E312 as the control, arm k = c2 and the stack E338; report only, no promotion of p or k, nothing selected or tuned on TEST |')
print("refusal with the row:", repr(TG.test_log_refusal(log + "\n" + row + "\n", "E343.t")))
print("refusal without the row:", repr(TG.test_log_refusal(log, "E343.t")))
