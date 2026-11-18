# 9. Robustness: the tests of the landed tree

**In short.** A system that is simple and fast must also be hard to break silently. The project's suite (`tools/maintenance/run_all_tests.sh`) runs the self-tests of the metric, data and model code (layer 0 of the project's seven layers); this chapter reports the run on the tree after the clean-up and the speed path landed (the speed path was removed after the run, see chapter 7), one full run. Every result is the printed result of that run; a test that did not run is written as not run.

Table: The suite on the landed tree (one full run of `tools/maintenance/run_all_tests.sh` by the lane chat, started 2026-11-18 06:41, log `temp/landing/suite.out`: 87 PASS, 1 FAIL; the tree was the landed one, including the speed path that was removed afterwards).

| group | what it guards | result of the run | notes |
|---|---|---|---|
| token head: cache, model, loss, decode, trainer | the head's function against its written spec; mutants (a wrong pair index, a swapped slice, a GT field in the prediction) must fail | PASS: `test_lane_tokcache`, `test_jtok`, `test_jtoken` | CPU, synthetic and real frames |
| token read-out and cost | dumps, sweep, cells, cards, paired scores; the time ensemble | PASS: `test_jtoken_eval`, `token_time_ens` (selftest ok, 8 mutants killed) | |
| live token producer and live training step (S4) | B' inside the training step; the parity with the cache | PASS: `test_lane_live`, `test_jlive_train` | the D120 champion was also re-read live: 81 metrics, 0 differing |
| speed path (removed after the run) | the same loss, gradients and weights as the plain path; the GPU loader's batches; the cache root | PASS: `test_jtoken_fast`, `test_lane_ptok_gpu`, `test_build_tokcache_root` | correct but not adopted (chapter 7); code and tests removed after the run, behind the tag `archive/token-speed-path-2026-11-18` |
| one arm tool | `jarm.py` selftest: the step chain, a failed step stops, a cut dump stops, the curve check, the ids of the card, the lock | PASS | |
| parallel scoring | `pmap.py`: identical to serial, order kept, nested call serial, exception raised | PASS | |
| lane evaluation suite | five checks, about 9 minutes | PASS: 5 / 5 in 334 s | |
| ground truth, targets and metrics | the metric self-tests with stored numbers, the target code (`test_qtargets.py`), the champion code against the original modules (exact, 24 frames x 2 seeds), golden digests | PASS | |
| report tools | `simplify_stat.py` selftest; `reporter.py`, `winner.py` selftests are run at the build | PASS (`simplify_stat`) | |
| autoresearch tools | launcher, git tag, ledger, schedule, decide (`autoresearch/tools/selftest.py`) | **FAIL once** in the run; PASS when re-run alone (flake under load, see below) | |

After the removal of the speed path the focused tests were run again on the committed tree: `test_jtoken`, `test_jtoken_eval`, `jarm` selftest, `pmap`, `simplify_stat` and `test_lane_tokcache`, all ok.

**Clean-HEAD check of the final tree** (commit `0b4a676165`, after the S2 hook left `jtoken.py` and the winners folder was re-exported; run by the lead's second chat, 2026-11-18 13:09 - 13:15, on a `git archive` copy without any data, CPU only). 7 of 7 counted suites pass: `test_jtoken`, `test_jtoken_eval`, `test_lane_tokcache`, the `jarm` selftest, the `pmap` selftest and the selftests of `check_git` and `commit_paths`. `test_jlive_train` and `test_lane_live` pass their selftests; their steps on real data need the data and are not counted. 0 of the 565 committed Python files import a module that exists only outside git. Of the 261 modules that the READMEs name, 242 import cleanly (the figures are from the report of the check, which kept no log in git); the other 19 need run data or figures that are never in git, are old loop-17 or preparation scripts with dead or external dependencies, or do work at import time. No row of the counted suites is red.

**What a green suite does not prove.** It proves that the code does what its tests say, on the fixtures and the real frames they use; it does not prove a number. Every number in this report comes from a stored card (`check_numbers.py` lists any decimal number of the text that no source holds), and the numbers of the champion were reproduced by the same read-out on two seeds.

**Known failures and flakes, named.** one test was red in the full run: the selftest of the autoresearch tools (launcher, git tag, ledger, schedule, decide) failed an assertion, "the command's pid and process group are in the record right after it started", a timing assertion of the launcher test. Re-run alone on the committed tree after the run (2026-11-18 10:29, other chats' jobs on the host) it passed (`autoresearch tools selftest passed`, exit 0). It is named a flake under host load: it polls for 10 seconds for the record of a started command, and the full run shared the host with training jobs and read-outs. The cause was not removed in this report; the test is not part of the lane code. The script's own exit code in the log reads 0 although a FAIL line stands in it (the last line of the script decides it): the log, not that code, is the result.

**sources** the run log of `run_all_tests.sh` (a result file of the run), the test files named above.
