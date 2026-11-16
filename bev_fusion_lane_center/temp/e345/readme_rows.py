"""rewrites my three README rows in place and writes the exact texts (and the tests/ row from HEAD + my clause) to temp/e345/readme_rows.md"""
import subprocess

R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
p = R + "evaluation/README.md"
lines = open(p, encoding="utf-8").read().split("\n")
probe = ("| `jprune_probe.py` | E345 (D99 step 0, part A; GPU side): what is lost when ONE unit of a stored junction system (E312, E324 s: frozen pyramid, junction maps head, pair head) is taken away without retraining. `units` (the table of the card: 45 units, 84 points, "
         "from the module tree), `sweep` (calibration on 201 TRAIN frames, kept in `calib.pt`; mean replacement of the lowest-importance channels / a skipped block / one attention head / an output channel / the paint inputs; the null twice, plumbing and random / reverse controls; "
         "per-frame counts, keypoint errors, pair AP and GPU peak memory of every pass over the 873 VAL frames in `sweep.json`; the dumps of the point sets named in `--score-points` into `runs/e345_p<NNN>_<model>/` (`point.json` written last, every file through a temporary file and "
         "`os.replace`); resumes; `--deterministic` is recorded and kept; the job waits for the GPU through `gpu_gate` for up to 12 h and exits with 75 instead of going on into a full GPU), `time` (seconds and GPU memory of one pass), `selftest`. `dump_frames` = `jends_eval.dump` "
         "without its file writing but NOT yet shared with it (TODO: `jends_eval.dump` should call it once no chain runs `jends_eval.py`; the sweep uses `forward_batches` + `frames_of`, and `tests/test_jprune_probe.py` compares `dump_frames` with a fresh `jends_eval.dump` on 16 frames). |")
table = ("| `jprune_table.py` | E345 (CPU side): `plan` (the point sets scored in full, from the seed-0 sweep; the four 'largest saved GMACs' slots count head units only; v once the handle-scale points are scored in full), `score` (the stored scorer `jends_eval.py score` on the "
         "point folders of `runs/`, tag e345<NNN>: every read-out for the two nulls, the two judged ones for the rest; an unfinished dump or a card without its counts is refused / reported), `boot` (the paired scene bootstrap of every scored point against the null through "
         "`jdiag.boot`, loss = null minus ablated, written next to the card; a FREE class needs the upper end <= 0.01), `table` (the two halves of the scenes summed from the per-frame counts, classes FREE / CHEAP / COSTLY / CRITICAL, the greedy cumulative curves, the sizes "
         "S_a / S_b / S_c, the controls, the full loss per read-out, `results.md` + `table.csv`); the job flow is in its docstring. Tests of all three: `tests/test_jprune_probe.py` (real E312 / E324 s models on the CPU, narrowed copies against FlopCounterMode, synthetic sweeps, "
         "stored cards, the point folder and its crash safety; 29 mutants of the CPU side and the probe's file / gate rules were checked to fail; about 7 minutes). |")
out = []
for i, ln in enumerate(lines):
    if ln.startswith("| `jprune_probe.py` |"):
        ln = probe
    elif ln.startswith("| `jprune_table.py` |"):
        ln = table
    out.append(ln)
open(p, "w", encoding="utf-8").write("\n".join(out))
head = subprocess.run(["git", "-C", R, "show", "HEAD:bev_fusion_lane_center/evaluation/README.md"], capture_output=True, text=True).stdout.split("\n")
trow = [ln for ln in head if ln.startswith("| `tests/` |")][0]
new_trow = trow.rstrip(" |") + "; `test_jprune_probe.py` (E345: the pruning probe, its cost table and its CPU side on the real E312 / E324 s models, narrowed copies, synthetic sweeps and stored cards; CPU, about 7 minutes). |"
cost = [ln for ln in lines if ln.startswith("| `jprune_cost.py` |")][0]
open(R + "temp/e345/readme_rows.md", "w", encoding="utf-8").write("ROW jprune_probe.py (new)\n" + probe + "\n\nROW jprune_cost.py (new, unchanged)\n" + cost + "\n\nROW jprune_table.py (new)\n" + table + "\n\nROW tests/ (HEAD text + the appended clause; NOT edited in place because another chat has uncommitted hunks in this row)\n" + new_trow + "\n")
print("ok")
