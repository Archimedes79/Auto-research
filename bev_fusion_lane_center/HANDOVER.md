# HANDOVER - 2026-11-12 17:00 (computer shut down at ~17:20 on Mirko's order)

New chat: start in the repository folder, say "Read HANDOVER.md and continue". Rewrite at every milestone (<= 3 KB). Mirko reads little: plain terms. D57 + D61: lane first. GPU only in his windows (nights 22-07, Mon-Fri 08:30-17:30). D60: Mirko away -> work alone. No TEST, push, mail without his word. Lane project is mine (D80). **Stay differentiable (D55); GaussianCaR features are the foundation (D50).**

## Where we are (nothing is running)
Report (private page, republish the same file path `results/Junction.report/index.html` with root + files list; builder `temp/jr/build_junction_report.py --media`): https://claude.ai/artifact/KpXQ2mrXkDjUtwBdJxFXeA (version 6).
**Final system = E312** (cards E285-E314, all committed, nothing pushed): E307 maps (end heat sigma 1.0 m: kp F1 0.72 vs 0.63) + pair head `--e2e --scales` (E305, E303) + connector ends snapped to the delivered pieces at p >= 0.1 (E311, `jends_eval.py score --snap`): topo F1 0.6825 / 0.6864 vs the plain graph 0.6259 / 0.6249 (+0.057 / +0.062; the pair head alone from the keypoints: +0.005), lanes found +0.11 / +0.13, junction F1 +0.05 / +0.07, maneuver coverage 52.5 % -> 70.7 % (`evaluation/jmaneuver.py`). Planner variant: + `--driven-w 2` (E298 KEEP: driven pair recall 0.50 / 0.51 vs 0.37 / 0.40, topo F1 unchanged). TIE and kept: E313 / E314 (sigma 0.7: more lanes, fewer isolated ends), E308, E287, E303 (scales +0.012 on the snapped read-out). Dropped / not run: E304 (stopped), E306, E302, jj (stopped at step 1000, ckpt kept).

## Code state
Committed: `evaluation/{jmaneuver,jdiag}.py` (new, reviewed), `qends_eval.py` / `jends_eval.py` (`--snap` and refinements, one card file per read-out; critic passes done), README + `run_gates.sh` rows; fast gates in `temp/gates_final.log`. Any run that builds heat targets needs `BFLC_SIG` (1.0 for the E307 maps, 0.7 for E313); dumps one at a time (flock `/tmp/jends_dump.lock`, host RAM).

## Next
1. For Mirko: D4, D5, soft step E279, TEST ONCE for the finished design (E312; his recorded OK and a log row), champion promotion / PROBLEM.md (protected: a decision), push, papers.csv merge, the full `GtGenerator selftest`, scratch `temp/gt_l6*` (delete only via `package.py`).
2. Ideas: cache the fitted handle scales (training 1.9 s / step is the fit), longer training / more seeds of E312, E298 + E312 jointly, road-level SD map counts (E291), Matrix-NMS layer (E292), time (E299).
