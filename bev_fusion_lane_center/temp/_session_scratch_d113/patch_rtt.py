import os
root = os.environ["BFLC_ROOT"]
p = os.path.join(root, "tools", "maintenance", "run_token_tests.sh")
s = open(p, encoding="utf-8").read()
i0 = s.index("# The tests of the token champion")
i1 = s.index("cd \"$(dirname")
hdr = '''# The LEGACY token tests that `run_all_tests.sh` does not list on purpose (since 2026-11-18 its last section lists the tests of the live champion path): the closed S2 / S3 blocks (E368 past, E374 map: both TIE by their written rules), the E354 recursive state, and the lane-suite selftest. Three at a time, niced, 2 threads each; one PASS / FAIL line per file, the last 30 lines of every output in the log folder.
# About 80 minutes on a shared box (test_jpast_train ~50 min under load, test_jmap_train ~44, test_lane_map ~37, test_lane_past ~27, test_jpast / test_jmap ~15 / 12, test_lane_suite ~9, test_jrecur ~2; CPU only, no GPU job needed). Last run 2026-11-17 / 18: all 8 PASS.
# usage (from the problem folder): wsl bash tools/maintenance/run_token_tests.sh [LOGDIR]      LOGDIR default temp/token_tests (scratch, ignored by git); exit 1 when any test failed.
'''
s = s[:i0] + hdr + s[i1:]
a = "printf '%s\n' test_jtoken_eval.py test_jtoken.py test_jmap_train.py test_jpast_train.py test_jlive_train.py test_lane_live.py test_lane_map.py test_jtok.py test_jtok2.py test_jrecur.py test_lane_past.py test_lane_tokcache.py test_jpast.py test_jmap.py test_build_tokcache_root.py test_lane_suite.py test_token_readout.py test_token_time_ens.py test_jtoken_fast.py test_lane_ptok_gpu.py"
assert s.count(a) == 1
s = s.replace(a, "printf '%s\n' test_jpast_train.py test_jmap_train.py test_lane_map.py test_lane_past.py test_jpast.py test_jmap.py test_lane_suite.py test_jrecur.py")
open(p, "w", encoding="utf-8").write(s)
# the readme blob
p2 = os.path.join(root, "temp", "_session_scratch_d113", "readme_tools_head2.md")
print("ok")
