#!/usr/bin/env python3
"""E347 scratch (WSL system python3): mutation check of the staged files. For every mutation: copy the stage tree to temp/_session_scratch_e347/mut/<id>, change ONE place (the old text must occur exactly once), run the test that guards the rule
through run_staged.sh with BFLC_STAGE pointing at the copy: the mutation is KILLED when the test fails, SURVIVED when it passes. Writes temp/_session_scratch_e347/mutation_report.txt.

    wsl python3 temp/e347/mutate.py [id ...]        (no id: all)
"""
import os
import re
import shutil
import subprocess
import sys

R = "/mnt/c/Auto-research/bev_fusion_lane_center"
STAGE = f"{R}/temp/_session_scratch_e347/stage"
MUT = f"{R}/temp/_session_scratch_e347/mut"
T2 = "training/grid/test_jmaps_r2.py"
TS = "training/grid/test_jmaps_simplify.py"
TE = "evaluation/jmaps_eval.py"

# (id, file, regex of the old text (must match once), replacement, test file, test args, what the mutation breaks)
M = [
    ("m01_offgrid_end_clipped_into_map", "training/grid/jgt.py", r"onehot\[k, cells\[inside, 0\], cells\[inside, 1\]\] = 1\.0", "onehot[k, cells[:, 0].clip(0, n - 1), cells[:, 1].clip(0, n - 1)] = 1.0", T2, ["--only", "end_cells"], "an end off the map is put into the border cell"),
    ("m02_round_cell", "training/grid/jgt.py", r"cells = np\.floor\(\(np\.asarray\(E, float\)\[:, :2\] \+ half\) / res\)", "cells = np.round((np.asarray(E, float)[:, :2] + half) / res)", T2, ["--only", "end_cells"], "the cell of an end by rounding, not by floor"),
    ("m03_exact_ignored", "training/grid/jgt.py", r"    if raster_exact\(\) if exact is None else exact:", "    if raster_exact():", T2, ["--only", "raster_2m"], "fill_poly ignores exact=True: the 2 m zone is the half-cell shifted legacy fill"),
    ("m04_zone_not_exact", "training/grid/qjunction.py", r"JG\.fields\(jg, n=N2, res=RES2, exact=True\) if r2", "JG.fields(jg, n=N2, res=RES2) if r2", T2, ["--only", "targets"], "the r2 data path rasters zone / core with the legacy fill"),
    ("m05_end_keys_swapped", "training/grid/qjunction.py", r't\["jend"\], t\["joff"\], t\["joffm"\] = JG\.end_cells', 't["joff"], t["jend"], t["joffm"] = JG.end_cells', T2, ["--only", "targets"], "one-hot and offset target swapped"),
    ("m06_offset_mean_all_cells", "training/grid/jmaps.py", r"\.sum\(\) / m\.sum\(\)\.clamp_min\(1\.0\)\n\n\ndef jmaps_loss", ".sum() / m.numel()\n\n\ndef jmaps_loss", T2, ["--only", "loss"], "the offset L1 averaged over all cells, not over the end cells"),
    ("m07_offset_unmasked", "training/grid/jmaps.py", r"\(\(pred - t\)\.abs\(\) \* m\.repeat_interleave\(2, dim=1\)\)\.sum\(\)", "(pred - t).abs().sum()", T2, ["--only", "loss"], "the offset L1 at every cell, not only at the end cells"),
    ("m08_off_weight", "training/grid/jmaps.py", r"OFF_W = 1\.0", "OFF_W = 0.5", T2, ["--only", "loss"], "the weight of the offset term"),
    ("m09_onehot_branch_off", "training/grid/jmaps.py", r'    if out\.get\("onehot", False\):', "    if False:", T2, ["--only", "loss"], "the one-to-one arm trains with the BCE + dice of the control"),
    ("m10_focal_norm", "training/grid/jmaps.py", r"max\(float\(je\[:, k\]\.sum\(\)\), 1\.0\)", "1.0", T2, ["--only", "loss"], "the focal loss not normalised by the number of ends"),
    ("m11_keypoint_no_offset", "training/grid/jmaps.py", r'    off = out\["j1"\]\.shape\[1\] >= NCH2', "    off = False", T2, ["--only", "keypoints"], "the read-out ignores the offset"),
    ("m12_peak_override_ignored", "training/grid/jmaps.py", r'    if \(not out\.get\("onehot", False\)\) if peak is None else peak:', '    if not out.get("onehot", False):', T2, ["--only", "keypoints"], "keypoints(peak=...) cannot switch the read-out"),
    ("m13_jends_peak_ignored", "training/grid/jends.py", r"            if self\.peak:", "            if True:", T2, ["--only", "pair_head"], "JEnds(peak=False) still has the local-maximum step"),
    ("m14_jends_offset_ignored", "training/grid/jends.py", r"            if self\.offset:", "            if False:", T2, ["--only", "pair_head"], "JEnds(offset=True) still takes the soft-argmax"),
    ("m15_softargmax_res", "training/grid/jends.py", r"x = c\[i0\] \+ \(wn \* dx\[None, :, None\] \* self\.res\)\.sum\(1\)", "x = c[i0] + (wn * dx[None, :, None] * RES).sum(1)", T2, ["--only", "pair_head"], "the soft-argmax in the pair head moves by 1 m cells on the 2 m grid"),
    ("m16_jends_centres_1m", "training/grid/jends.py", r"        c = _centres\(dev, self\.n1, self\.res\)", "        c = _centres(dev)", T2, ["--only", "pair_head"], "the cell centres of the pair head are the 1 m centres"),
    ("m17_gt_cell_1m", "training/grid/jends.py", r"i = \(\(ends\[\.\.\., :2\] \+ HALF\) / self\.res\)\.floor", "i = ((ends[..., :2] + HALF) / RES).floor", T2, ["--only", "pair_head"], "the GT ends are injected into the 1 m cells"),
    ("m18_model_peak_from_ctx", "training/train/jends_train.py", r'peak="onehot" not in mods\)', "peak=True)", T2, ["--only", "glue"], "JEndModel gives the one-to-one head a peak step"),
    ("m19_model_grid_from_ctx", "training/train/jends_train.py", r'n1=50 if "r2" in mods else 100', "n1=100", T2, ["--only", "glue"], "JEndModel builds the pair head of a 2 m head for 100 cells"),
    ("m20_dup_counts_matched", "evaluation/jmaps_eval.py", r"return int\(\(near & ~claimed\)\.sum\(\)\)", "return int(near.sum())", TE, ["selftest"], "matched keypoints count as surplus duplicates"),
    ("m21_gate_one_threshold", "evaluation/jmaps_eval.py", r'sw\[typ\]\["best_thr"\] if thr is None', 'sw["ent"]["best_thr"] if thr is None', TE, ["selftest"], "both kinds read at the entry's best threshold"),
    ("m22_crowded_no_heading", "evaluation/jmaps_eval.py", r"for _, j, _ in match\(pred, gt, 3\.0, use_ang\)\[0\]:", "for _, j, _ in match(pred, gt, 3.0, False)[0]:", TE, ["selftest"], "the crowded recall ignores the heading"),
    ("m23_head_f1_not_2m_features", "training/grid/jmaps.py", r"out = dict\(j1=self\.head2\(c\), f1=c, r2=True\)", 'out = dict(j1=self.head2(c), f1=self.p2(P["feat2"]), r2=True)', TS, [], "f1 of the 2 m head is not the output of the context blocks"),
    ("m24_offset_term_off", "training/grid/jmaps.py", r"    if r2:\s+# E347: the offset of the end within its 2 m cell", "    if False:  # E347: the offset of the end within its 2 m cell", T2, ["--only", "loss"], "no offset loss"),
    ("m28_dropped_counts_members", "evaluation/jend_grid_stats.py", r"dropped=int\(\(cnt - 1\)\.sum\(\)\)", "dropped=int(cnt[cnt > 1].sum())", "evaluation/jend_grid_stats.py", ["selftest"], "the ends beyond the first per cell counted as all members of shared cells"),
    ("m29_adjacent_cells_wide", "evaluation/jend_grid_stats.py", r"np\.where\(cheb == 1, 1, 2\)", "np.where(cheb <= 2, 1, 2)", "evaluation/jend_grid_stats.py", ["selftest"], "an end two cells away counted as adjacent"),
    # round 2 (offset supervision within 3 m, the finer gate grid, the scorer tests, --ckpt, the sigma guard)
    ("m30_off_near_1m", "training/grid/jgt.py", r"OFF_NEAR = 3\.0 ", "OFF_NEAR = 1.0 ", T2, ["--only", "end_cells,targets"], "the offset is trained within 1 m of an end, not 3 m"),
    ("m31_offset_to_first_end", "training/grid/jgt.py", r"        a = d2\.argmin\(2\)", "        a = np.zeros(d2.shape[:2], int)", T2, ["--only", "end_cells,targets"], "the offset target points to the first end, not the nearest"),
    ("m32_loss_mask_is_onehot", "training/grid/jmaps.py", r'batch\["joffm"\]\.to\(dev\)\.float\(\)', 'batch["jend"].to(dev).float()', T2, ["--only", "loss"], "the offset loss uses the one-hot cells, not the 3 m mask"),
    ("m33_offset_per_kind_norm", "training/grid/jmaps.py", r"    return \(\(pred - t\)\.abs\(\) \* m\.repeat_interleave\(2, dim=1\)\)\.sum\(\) / m\.sum\(\)\.clamp_min\(1\.0\)", "    return sum(((pred[:, 2 * k: 2 * k + 2] - t[:, 2 * k: 2 * k + 2]).abs().sum(1) * m[:, k]).sum() / m[:, k].sum().clamp_min(1.0) for k in range(2))", T2, ["--only", "loss"], "one normaliser per kind instead of one over all masked cells"),
    ("m34_gate_grid_coarse", "evaluation/jmaps_eval.py", r"GATE_THR = tuple\(round\(0\.05 \* i, 2\) for i in range\(1, 20\)\)", "GATE_THR = THR_SWEEP", TE, ["selftest"], "the gate grid is the old 0.2 ... 0.7"),
    ("m35_edge_never", "evaluation/jmaps_eval.py", r'thr is None and r\["thr"\] in \(', 'False and r["thr"] in (', TE, ["selftest"], "an optimum at the end of the grid is not flagged"),
    ("m36_readouts_swapped", "evaluation/jmaps_eval.py", r"return dict\(ours=JM\.keypoints\(out, thr=t\), ours_alt=JM\.keypoints\(out, thr=t, peak=jm\.onehot\)\)", "return dict(ours=JM.keypoints(out, thr=t, peak=jm.onehot), ours_alt=JM.keypoints(out, thr=t))", TE, ["selftest"], "`ours` and `ours_alt` swapped"),
    ("m37_readout_names_swapped", "evaluation/jmaps_eval.py", r'return dict\(ours="threshold" if jm\.onehot else "peak", ours_alt="peak" if jm\.onehot else "threshold"\)', 'return dict(ours="peak" if jm.onehot else "threshold", ours_alt="threshold" if jm.onehot else "peak")', TE, ["selftest"], "the read-out names of the rule are inverted"),
    ("m38_best_by_f1_1_5", "evaluation/jmaps_eval.py", r'best_thr=max\(rows, key=lambda r: r\["f1"\]\)', 'best_thr=max(rows, key=lambda r: r["f1_tol1_5"])', TE, ["selftest"], "the best threshold chosen by the 1.5 m F1"),
    ("m39_dup_boundary", "evaluation/jmaps_eval.py", r"\.query_pairs\(r, ", ".query_pairs(r - 0.001, ", TE, ["selftest"], "a copy exactly 2 m away is not a duplicate"),
    ("m40_ckpt_wrong_key", "evaluation/jmaps_eval.py", r'return ck\["model"\], cfg', 'return ck["state"], cfg', TE, ["selftest"], "--ckpt reads the wrong key of ckpt.pt"),
    ("m43_ckpt_weights_only", "evaluation/jmaps_eval.py", r', weights_only=False\)', ')', TE, ["selftest"], "--ckpt loads ckpt.pt with the default weights_only (refuses the RNG states)"),
    ("m41_sigma_guard_off", "training/train/jends_train.py", r"        assert got == want, ", "        assert True, ", T2, ["--only", "glue"], "the pair stage of a 2 m head accepts another BFLC_SIG"),
    ("m42_names_without_joffm", "training/data/lane_gt_data.py", r'"joff", "joffm"\)', '"joff")', T2, ["--only", "targets"], "the collate does not stack the offset mask"),
    # the wiring of the data path and the trainers (real frames, CPU, the optimiser loop stubbed): temp/e347/smoke_trainers.py --quick
    ("m25_data_r2_not_passed", "training/data/lane_gt_data.py", r"QJ\.junction_targets\(g, paint, r2=self\.r2\)", "QJ.junction_targets(g, paint)", f"{R}/temp/e347/smoke_trainers.py", ["--quick"], "QData does not hand r2 to the targets"),
    ("m26_jmaps_train_no_r2_data", "training/train/jmaps_train.py", r"    LD\.QData\.DEFAULT_R2 = a\.r2 ", "    LD.QData.DEFAULT_R2 = False ", f"{R}/temp/e347/smoke_trainers.py", ["--quick"], "jmaps_train.py builds the loader without the 2 m targets"),
    ("m27_jends_train_no_r2_data", "training/train/jends_train.py", r'    LD\.QData\.DEFAULT_R2 = "r2" in JM\.ctx_mods\(ck\["args"\]\.get\("ctx", "dil"\)\)', "    LD.QData.DEFAULT_R2 = False", f"{R}/temp/e347/smoke_trainers.py", ["--quick"], "jends_train.py builds the loader without the 2 m targets"),
]


def run(mid, f, pat, rep, test, targs, what):
    d = f"{MUT}/{mid}"
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(STAGE, d)
    s = open(f"{d}/{f}", encoding="utf-8").read()
    s2, n = re.subn(pat, lambda _m: rep, s)
    assert n == 1, f"{mid}: the pattern matched {n} times"
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s2)
    env = dict(os.environ, BFLC_STAGE=d, E347_FAST="1")
    p = subprocess.run(["bash", f"{R}/temp/e347/run_staged.sh", test if test.startswith("/") else f"{d}/{test}", *targs], env=env, capture_output=True, text=True)
    tail = (p.stdout.strip().splitlines() or [""])[-1][:90] + " | " + ([l for l in p.stderr.strip().splitlines() if "Error" in l or "assert" in l.lower()] or [""])[-1][:140]
    shutil.rmtree(d)
    return p.returncode != 0, tail


if __name__ == "__main__":
    want = sys.argv[1:]
    rows = []
    for mid, f, pat, rep, test, targs, what in M:
        if want and mid not in want:
            continue
        killed, tail = run(mid, f, pat, rep, test, targs, what)
        rows.append(f"{'KILLED  ' if killed else 'SURVIVED'} {mid:30s} {what}  [{os.path.basename(test)} {' '.join(targs)}]  {tail if killed else ''}")
        print(rows[-1], flush=True)
    open(f"{R}/temp/_session_scratch_e347/mutation_report.txt", "w" if not want else "a").write("\n".join(rows) + "\n")
