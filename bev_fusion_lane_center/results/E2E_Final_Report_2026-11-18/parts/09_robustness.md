# 9. Robustness: the tests of the landed tree

**In short.** A system that is simple and fast must also be hard to break silently. The project's suite (`tools/maintenance/run_all_tests.sh`) runs the self-tests of the metric, data and model code (layer 0 of the project's seven layers); this chapter reports the run on the tree after the clean-up and the speed path landed, one full run, with the GPU tests alone. Every result is the printed result of that run; a test that did not run is written as not run.

Table: The suite on the landed tree (one full run; CPU tests any time, GPU tests alone on the GPU).

| group | what it guards | result of the run | notes |
|---|---|---|---|
| token head: model, loss, decode, evaluator | the head's function against its written spec; mutants (a wrong pair index, a swapped slice, a GT field in the prediction) must fail | PENDING | `test_jtoken.py`, `test_jtoken_eval.py` |
| speed path | the same loss, gradients and 20 smoke steps as the plain path (to 1e-5 loss, 2e-5 gradients, 3e-6 weights); the GPU loader's batches equal the plain loader's bit for bit; the epoch order | PENDING | `test_jtoken_fast.py`, `test_lane_ptok_gpu.py`, `test_build_tokcache_root.py` |
| GPU tests (bf16, no host sync, compile) | what the CPU cannot show; run with `JTF_GPU=1`, alone on the GPU | PENDING | prints `gpu ok` |
| one arm tool | `jarm.py` selftest: the step chain, a failed step stops, a cut dump stops, the curve check, the ids of the card, the lock | PENDING | `jarm.py selftest` |
| parallel scoring | `pmap.py`: identical to serial, order kept, nested call serial, exception raised | PENDING | `evaluation/pmap.py` |
| ground truth and targets | the generator's self-tests and pinned digests; the target code (`test_qtargets.py`: one bound is a data property after the GT merge, 1.04 % against the bound 1.1 %) | PENDING | `GtGenerator` selftest, `test_qtargets.py` |
| report tools | `reporter.py selftest`, `winner.py selftest`, `simplify_stat.py selftest` | PENDING | |

**What a green suite does not prove.** It proves that the code does what its tests say, on the fixtures and the real frames they use; it does not prove a number. Every number in this report comes from a stored card (`check_numbers.py` lists any decimal number of the text that no source holds), and the numbers of the champion were reproduced by the same read-out on two seeds.

**Known failures and flakes, named.** PENDING: the list of the run (a test that is red is listed with its cause; a flake under host load is listed as such and passed alone).

**sources** the run log of `run_all_tests.sh` (a result file of the run), the test files named above.
