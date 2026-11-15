"""E347: write the control's numbers (E332 arm a, scored with the E347 scorer, `runs/e332_ma_s<S>/jmaps_val_legacy.json`) into the card BEFORE any arm read-out (card section 2b (e)).
usage: python3 temp/_session_scratch_s292/e347_control_numbers.py   (plain python; prints the text and inserts it after the 'Launch:' bullet of section 2b)"""
import io
import json
import time

p = "experiments/E347_ends-on-the-2m-node-grid/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
assert "Control numbers (written" not in s
rows = []
for S in (0, 1):
    d = json.load(open(f"runs/e332_ma_s{S}/jmaps_val_legacy.json"))
    st = json.load(open(f"runs/e332_ma_s{S}/jmaps_val.json"))
    g = d["gate"]
    for name, key in (("peak read-out (the control's own rule), at its best threshold", ("ours", "at_best")), ("peak read-out at 0.3", ("ours", "at_thr0_3")),
                      ("threshold alone (the other read-out), at its best threshold", ("ours_alt", "at_best")), ("threshold alone at 0.3", ("ours_alt", "at_thr0_3"))):
        r = g[key[0]][key[1]]
        rows.append((S, name, r["thr"]["ent"], r["thr"]["ext"], r["f1_tol3"], r["f1_tol1_5"], r["crowded_recall"]["ent"], r["crowded_recall"]["ext"], r["dup_share"], r["edge"]["ent"] or r["edge"]["ext"]))
    last = d["frames"]
txt = "- **Control numbers (written 2026-11-15 TIME, BEFORE any arm read-out; `E347.c_score_s0` / `_s1`: the E347 scorer on the stored E332 arm a, raster legacy, VAL, %d frames, threshold grid 0.05 - 0.95; `runs/e332_ma_s<S>/jmaps_val_legacy.json`, key `gate`; F1 = mean of entry and exit; the numbers at 0.3 reproduce the stored `jmaps_val.json`):**\n\n" % last
txt += "| seed | read-out | best thr ent / ext | keypoint F1 at 3 m | F1 at 1.5 m | crowded entry recall | crowded exit recall | surplus-duplicate share | EDGE |\n|---|---|---|---|---|---|---|---|---|\n"
for S, name, te, tx, f3, f15, ce, cx, ds, ed in rows:
    txt += f"| {S} | {name} | {te:.2f} / {tx:.2f} | {f3:.4f} | {f15:.4f} | {ce:.4f} | {cx:.4f} | {ds:.4f} | {'yes' if ed else 'no'} |\n"
txt += "\nThe stage-1 clauses are read against the first row of each seed (the control's own rule at its best threshold) and the arms at THEIR best thresholds on the same grid.\n"
txt = txt.replace("TIME", time.strftime("%H:%M"))
print(txt)
anchor = "- **Launch:** `bash temp/e347/start_arm.sh"
assert s.count(anchor) == 1
s = s.replace(anchor, txt + "\n" + anchor)
io.open(p, "w", encoding="utf-8", newline="").write(s)
