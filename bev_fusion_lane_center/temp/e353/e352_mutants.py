"""mutation check of tests/test_jprune_b.py (scratch): each mutant is applied to evaluation/jprune_b.py, the named tests run, the file is restored; KILLED = a test fails. Run in WSL: python3 temp/e353/e352_mutants.py [names]"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
F = "evaluation/jprune_b.py"
MUTANTS = [
    ("edit_not_applied_to_B", "    shim, wb = PP.ablate(Shim(qg), edits, taps)", "    shim, wb = PP.ablate(Shim(qg), [], taps)", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
    ("edit_not_applied_to_the_junction_model", "    mj, wj = PP.ablate(jm, edits, taps)", "    mj, wj = PP.ablate(jm, [], taps)", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
    ("random_is_ranked", "        out += group_edits(model, u, f, calib, how, rng) if groups else PP.edits_for(u, f, calib, how, rng)", "        out += group_edits(model, u, f, calib, how, rng) if groups else PP.edits_for(u, f, calib, \"rank\", None)", ["points_ladder_counts_and_reference_channels"]),
    ("random_seed_without_the_unit", "RandomState(1000 * draw + sum(map(ord, unit)))", "RandomState(1000 * draw)", ["points_ladder_counts_and_reference_channels"]),
    ("group_ranking_reversed", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), stable=True)", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), descending=True, stable=True)", ["points_ladder_counts_and_reference_channels"]),
    ("premise_not_asserted", "    assert len(keys) == 57 and all(torch.equal(sb[k], sj[\"jmodel.\" + k]) for k in keys), \"B's pyramid", "    assert True or all(torch.equal(sb[k], sj[\"jmodel.\" + k]) for k in keys), \"B's pyramid", ["pyramid_premise_and_shim"]),
    ("overwrite_allowed", "    if os.path.exists(os.path.join(folder, \"point.json\")):\n        raise SystemExit(f\"{folder} is complete already: a point folder is never overwritten\")\n    os.makedirs", "    os.makedirs", ["point_folder_files_hashes_and_refusals"]),
    ("link_to_B_checkpoint", "os.symlink(os.path.abspath(os.path.join(ROOT, \"runs\", j, \"model.pt\")), link)", "os.symlink(os.path.abspath(os.path.join(ROOT, \"runs\", b, \"model.pt\")), link)", ["point_folder_files_hashes_and_refusals"]),
    ("partial_dump_into_runs", "        assert not a.max or a.out, \"--max writes a partial dump: give --out (a folder outside runs/)\"\n", "", ["partial_dump_cannot_land_in_runs"]),
    ("score_line_without_base", " --base $R/runs/e273_p_s{seed}/model.pt", "", ["scoring_lines_pass_the_stored_checks"]),
    ("score_line_wrong_tag", "--spec close:1.5@0.5 --tag {name}\"", "--spec close:1.5@0.5 --tag {name}x\"", ["scoring_lines_pass_the_stored_checks"]),
    ("comparison_blind_to_nodes", "        r[\"nodes_only_in_ref\"] += len(um_a)\n        r[\"nodes_only_in_new\"] += len(um_b)\n", "", ["dump_comparison_catches_a_changed_dump"]),
    ("comparison_ignores_links", "        r[\"links_only_in_ref\"] += len(one_a)\n        r[\"links_only_in_new\"] += len(one_b)\n", "", ["dump_comparison_catches_a_changed_dump"]),
    ("dump_without_dead_hook_check_and_wrong_order", "    assert [f[\"row\"] for f in frames] == rows[:n], \"the junction frames are not in the order of B's loader\"", "    frames = frames[::-1]", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
]


def main():
    want = sys.argv[1:]
    res = {}
    for name, old, new, tests in MUTANTS:
        if want and name not in want:
            continue
        p = os.path.join(ROOT, F)
        src = open(p, encoding="utf-8").read()
        if src.count(old) != 1:
            print(f"SKIPPED  {name}: the text to mutate is found {src.count(old)} times", flush=True)
            continue
        shutil.copyfile(p, p + ".orig_mut")
        t0 = time.time()
        try:
            open(p, "w", encoding="utf-8").write(src.replace(old, new))
            r = subprocess.run(["nice", "bash", "tools/env/_py14.sh", "tests/test_jprune_b.py", *tests], cwd=ROOT, capture_output=True, text=True)
        finally:
            shutil.move(p + ".orig_mut", p)
        killed = r.returncode != 0
        tail = [x for x in (r.stdout + r.stderr).strip().split("\n") if x][-1][:160]
        res[name] = killed
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name} ({time.time() - t0:.0f} s): {tail}", flush=True)
        assert open(p, encoding="utf-8").read() == src, "restore failed"
    print("survivors:", [n for n, k in res.items() if not k])


if __name__ == "__main__":
    main()
