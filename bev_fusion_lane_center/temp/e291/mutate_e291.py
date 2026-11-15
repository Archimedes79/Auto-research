#!/usr/bin/env python3
"""E291 scratch (WSL system python3): mutation check of the staged files and of training/grid/jprior.py. For every mutation: copy the stage tree to temp/_session_scratch_e291/mut/<id> (with the real jprior.py inside, so that the finder serves the mutated copy),
change ONE place (the old text must match exactly once), run the test that guards the rule through run_staged.sh with BFLC_STAGE pointing at the copy: KILLED when the test fails, SURVIVED when it passes. The report goes to temp/_session_scratch_e291/mutation_report.txt.

    wsl bash -c "cd /mnt/c/Auto-research/bev_fusion_lane_center && python3 temp/e291/mutate_e291.py [id ...]"        (no id: all; `cheap`: the ones whose test takes under a minute)
"""
import os
import re
import shutil
import subprocess
import sys

R = "/mnt/c/Auto-research/bev_fusion_lane_center"
SCR = f"{R}/temp/_session_scratch_e291"
STAGE, MUT = f"{SCR}/stage", f"{SCR}/mut"
JPR = "training/grid/jprior.py"
JS = ("jprior.py", [])                                                  # the selftest of the module
TS = lambda *sec: ("test_e291_staged.py", ["--only", ",".join(sec)])  # noqa: E731
M = [
    # (id, file, regex (once), replacement, (test, args), what it breaks)
    ("p01_radius_6", JPR, r"RADIUS, COS_MIN = 10\.0, 0\.5", "RADIUS, COS_MIN = 6.0, 0.5", JS, "the presence / association radius is 6 m, not 10"),
    ("p02_rank_reversed", JPR, r"lat_k > lat_i\[:, None, :\]\) \|", "lat_k < lat_i[:, None, :]) |", JS, "the lateral rank counts from the right"),
    ("p03_count_of_wrong_direction", JPR, r"n = torch\.where\(ex, arms\[\"n_out\"\]\[:, None, :\], arms\[\"n_in\"\]\[:, None, :\]\)", 'n = torch.where(ex, arms["n_in"][:, None, :], arms["n_out"][:, None, :])', JS, "an exit keypoint reads the entry count"),
    ("p04_injected_counted", JPR, r"& ok\[:, :, None\] & \(kind", "& (kind", JS, "found / rank also count a keypoint that is not a real peak"),
    ("p05_heading_gate_off", JPR, r"& \(cos >= COS_MIN\)", "& (cos >= -2.0)", JS, "an exit keypoint heading into the junction belongs to the arm"),
    ("p06_unknown_not_flagged", JPR, r"\(~known\)\.float\(\)", "(~has).float()", JS, "an unknown count is not flagged"),
    ("p07_wrong_rate_doubled", JPR, r"ni, no = \(_wrong\(ni, u\[6\]\) if u\[5\] < r else ni\)", "ni, no = (_wrong(ni, u[6]) if u[5] < 2 * r else ni)", JS, "the share of wrong counts is twice the rate"),
    ("p08_shift_at_rate_0", JPR, r"            if r > 0:                                            # a junction with no error", "            if True:                                             # a junction with no error", JS, "a junction with r = 0 is shifted and turned"),
    ("p09_wrong_may_equal_truth", JPR, r"cand = \[v for v in range\(MAX_WRONG \+ 1\) if v != n\]", "cand = [v for v in range(MAX_WRONG + 1)]", JS, "a 'wrong' count can be the true one"),
    ("p10_no_zero_start", JPR, r"        for p in \(self\.gate\[2\]\.weight, self\.gate\[2\]\.bias, self\.feat\.weight, self\.feat\.bias\):\n            nn\.init\.zeros_\(p\)", "        pass", JS, "the term is not zero at the start"),
    ("p11_positions_not_detached", JPR, r"@torch\.no_grad\(\)\ndef features", "def features", JS, "the prior path reaches the keypoint positions (no_grad off)"),
    ("p11b_no_detach_no_nograd", JPR, [r"@torch\.no_grad\(\)\ndef features", r"features\(pos\.detach\(\)\.float\(\), D\.detach\(\)\.float\(\)"], ["def features", "features(pos.float(), D.float()"], JS, "the prior path reaches the keypoint positions (both guards off: p11 alone is equivalent, either guard suffices)"),
    ("p12_junction_ignores_token", JPR, r"u = _u\(\"junction\", token, z\)", 'u = _u("junction", z)', JS, "the junction draw does not depend on the frame"),
    ("p13_partner_is_neighbour", JPR, r"s = tokens\.index\(token\) \+ n // 2", "s = tokens.index(token)", JS, "the partner of S is the next frame, not half a split away"),
    ("p14_pm1_below_zero", JPR, r"n \+ \(1 if sg < 0\.5 or n == 0 else -1\)", "n + (1 if sg < 0.5 else -1)", JS, "a count of 0 can become -1"),
    ("p15_junction_cases_overlap", JPR, r"u < spec\.drop \+ spec\.ujunc", "u < spec.ujunc", JS, "the junction without a map and the unknown junction are one set"),
    ("p16_absent_not_zero", JPR, r"torch\.where\(has, lat_i\.abs\(\) / 20\.0, z\)", "lat_i.abs() / 20.0", JS, "an absent keypoint carries a distance"),
    ("p17_ok_is_valid_only", JPR, r"features\(pos\.detach\(\)\.float\(\), D\.detach\(\)\.float\(\), kind, \(valid & real\)\.flatten\(1\), arms\)", "features(pos.detach().float(), D.detach().float(), kind, valid.flatten(1), arms)", TS("inject"), "the head counts the injected GT keypoints"),
    ("p18_single_assignment", JPR, r"mem = elig\.gather\(2, own\) &", "mem = (own == arm[:, :, None]) &", JS, "the keypoints of an arm are only the ones assigned to it (a keypoint near two arms counts for one)"),
    # the staged hunks
    ("s01_term_not_added", "training/grid/jends.py", r"            t = t \+ term\.to\(t\.dtype\)\n", "            t = t\n", TS("smoke"), "the prior term is computed but not added to the token"),
    ("s02_prior_built_first", "training/grid/jends.py", r"        self\.prior = JP\.PriorHead\(d=d\) if prior else None[^\n]*\n", "", TS("identity_random"), "the prior is not built (no such parameters)"),
    ("s03_prior_flag_ignored", "training/grid/jends.py", r"paint_ev=True, prior=False\):", "paint_ev=True, prior=True):", TS("identity_random"), "flag off builds the prior"),
    ("s04_trainer_without_prior_params", "training/train/jends_train.py", r"    ps = list\(model\.je\.parameters\(\)\)", "    ps = list(model.je.blocks.parameters()) + list(model.je.pair.parameters()) + list(model.je.scale.parameters()) + list(model.je.kp.parameters())", TS("smoke"), "the prior is outside the trainer's parameter set"),
    ("s05_sigma_assert_off", "training/train/jends_train.py", r"    assert abs\(sig - float\(ck\[\"args\"\]\.get\(\"sigma\", 1\.5\)\)\) < 1e-9,", "    assert True or abs(sig - 0) < 1e-9,", TS("bfsig"), "a launcher without BFLC_SIG is not stopped"),
    ("s06_clean_training_spec", "training/train/jends_train.py", r"JP\.Spec\.train\(a\.prior_noise\)", 'JP.Spec.train("0")', TS("smoke"), "--prior-noise mix trains on clean tables"),
    ("s07_readouts_share_table", "evaluation/jends_eval.py", r'b\["arms"\]\[r\]\.items\(\)', 'b["arms"][0].items()', TS("dump"), "every read-out of the pass uses the first table"),
    ("s08_dump_overwrites", "evaluation/jends_eval.py", r"os\.makedirs\(d, exist_ok=False\)", "os.makedirs(d, exist_ok=True)", TS("dump"), "a read-out folder is overwritten"),
    ("s09_data_arms_not_collated", "training/data/lane_gt_data.py", r"out\[\"arms\"\] = \[JPR\.collate\(\[o\[\"arms\"\]\[k\] for o in b\]\) for k in range\(len\(b\[0\]\[\"arms\"\]\)\)\]", 'out["arms"] = [JPR.collate([o["arms"][0] for o in b])] * len(b[0]["arms"])', TS("keys"), "the collate repeats the first table"),
    ("s10_partner_not_other_frame", "training/data/lane_gt_data.py", r"for tk in JPR\.partners\(token, self\._ptok\):", "for tk in [token]:", TS("shuffle"), "the S read-out is the frame's own table"),
]


def run(mid, f, pat, rep, test, what):
    d = f"{MUT}/{mid}"
    if os.path.exists(d):
        shutil.rmtree(d)
    shutil.copytree(STAGE, d)
    os.makedirs(f"{d}/training/grid", exist_ok=True)
    shutil.copy(f"{R}/{JPR}", f"{d}/{JPR}")
    s2 = open(f"{d}/{f}", encoding="utf-8").read()
    for pt, rp in zip(pat if isinstance(pat, list) else [pat], rep if isinstance(rep, list) else [rep]):               # a mutation of two guards at once: lists
        s2, n = re.subn(pt, lambda _m, rp=rp: rp, s2)
        assert n == 1, f"{mid}: the pattern matched {n} times"
    open(f"{d}/{f}", "w", encoding="utf-8", newline="\n").write(s2)
    env = dict(os.environ, BFLC_STAGE=d, E291_FAST="1")
    script = test[0] if test[0] != "test_e291_staged.py" else f"{R}/temp/e291/test_e291_staged.py"
    p = subprocess.run(["bash", f"{R}/temp/e291/run_staged.sh", script, *test[1]], env=env, capture_output=True, text=True)
    errs = [l for l in p.stderr.strip().splitlines() if "Error" in l or "assert" in l.lower()]
    tail = (errs or [""])[-1][:160]
    shutil.rmtree(d)
    return p.returncode != 0, tail


if __name__ == "__main__":
    want = sys.argv[1:]
    rows = []
    for mid, f, pat, rep, test, what in M:
        if want and mid not in want and not ("cheap" in want and test == JS):
            continue
        killed, tail = run(mid, f, pat, rep, test, what)
        rows.append(f"{'KILLED  ' if killed else 'SURVIVED'} {mid:34s} {what}  [{test[0]} {' '.join(test[1])}]  {tail if killed else ''}")
        print(rows[-1], flush=True)
    open(f"{SCR}/mutation_report.txt", "a").write("\n".join(rows) + "\n")
