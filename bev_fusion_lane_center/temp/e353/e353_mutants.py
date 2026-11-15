"""mutation check of test_jnarrow.py (scratch): each mutant is applied to the real file, the named tests run, the file is restored; a mutant is KILLED when a test fails. Run in WSL: python3 temp/e353/e353_mutants.py [names]"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID, TRAIN = "training/grid/jnarrow.py", "training/train/jnarrow_train.py"
MUTANTS = [
    ("wrong_channel_set_most_important_groups", GRID, "gone = sorted(torch.argsort(score, stable=True)[:n_rm].tolist())", "gone = sorted(torch.argsort(score, descending=True, stable=True)[:n_rm].tolist())", ["groups_are_the_lowest_importance_groups"]),
    ("wrong_channel_set_per_channel_probe_ranking", GRID, "            score = calib.s[tap].double().view(G, gs).sum(1)                           # summed importance of the channels of a group\n            gone = sorted(torch.argsort(score, stable=True)[:n_rm].tolist())\n            out[tap] = dict(removed=[c for g in gone for c in range(g * gs, (g + 1) * gs)], groups=[G - n_rm, gs])",
     "            ch = sorted(PP.pick(calib.s[tap], n_rm * gs).tolist())\n            out[tap] = dict(removed=ch, groups=[G - n_rm, gs])", ["groups_are_the_lowest_importance_groups"]),
    ("bias_fold_left_out", GRID, "        nr.bias.data.copy_(((r.bias.data.double() if r.bias is not None else 0.0) + fold).to(r.weight.dtype))", "        nr.bias.data.copy_(((r.bias.data.double() if r.bias is not None else 0.0) + 0 * fold).to(r.weight.dtype))", ["equality_with_the_hook_ablated_model", "border_difference_is_the_padding_term"]),
    ("fold_with_wrong_mean_zero_kernel_sum", GRID, "(r.weight.data[:, [o + c for c in removed]].double() * mean[removed].double().view(1, -1, 1, 1)).sum((1, 2, 3))", "(r.weight.data[:, [o + c for c in removed]].double() * mean[removed].double().view(1, -1, 1, 1)).sum((1, 2))[:, 0]", ["equality_with_the_hook_ablated_model"]),
    ("wrong_consumer_columns_sliced", GRID, "        cols = sorted(o + c for o in offs for c in removed)\n", "        cols = list(range(len(removed)))\n", ["equality_with_the_hook_ablated_model"]),
    ("producer_rows_of_the_wrong_channels", GRID, "new.weight.data.copy_(conv.weight.data[keep])", "new.weight.data.copy_(conv.weight.data[sorted(removed)[:len(keep)] if len(removed) >= len(keep) else keep])", ["equality_with_the_hook_ablated_model"]),
    ("group_norm_affine_not_sliced", GRID, "    ng.weight.data.copy_(gn.weight.data[keep])", "    ng.weight.data.copy_(gn.weight.data[:len(keep)])", ["equality_with_the_hook_ablated_model"]),
    ("mac_count_other_channel_set", GRID, "idx=tuple(v[\"removed\"]), val=None if calib is None else", "idx=tuple(v[\"removed\"][:-4]), val=None if calib is None else", ["macs_and_parameters_equal_the_cost_table"]),
    ("one_group_less_removed", GRID, "gs, n_rm = C // G, int(f * G + 0.5)", "gs, n_rm = C // G, int(f * G + 0.5) - 1", ["groups_are_the_lowest_importance_groups", "macs_and_parameters_equal_the_cost_table"]),
    ("loader_ignores_the_spec", GRID, "    if not a.get(\"narrow\", {}).get(\"taps\"):\n        return _STORED_LOAD(run, dev)", "    if True:\n        return _STORED_LOAD(run, dev)", ["spec_round_trip_and_loaders"]),
    ("spec_loses_the_group_count", GRID, "assert [gn.num_groups, gn.num_channels // gn.num_groups] == t[\"groups\"], (tap, gn, t)", "pass", ["spec_round_trip_and_loaders"]),
    ("narrowed_dump_through_the_stored_loader", "training/train/jnarrow_eval.py", "    JEV.load_model = JN.load_model", "    pass", ["dump_wrapper_and_score_check", "timing_machinery_on_a_narrowed_toy"]),
    ("pyramid_left_trainable", TRAIN, "    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters())", "    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters()) + list(model.jmodel.pyr.parameters())", ["frozen_set_and_checks_at_the_start"]),
    ("node_head_left_trainable", TRAIN, "    for p in model.parameters():\n        p.requires_grad_(False)\n    for p in ps:", "    for p in model.parameters():\n        p.requires_grad_(False)\n    for p in model.nodes.parameters():\n        p.requires_grad_(True)\n    for p in ps:", ["frozen_set_and_checks_at_the_start"]),
    ("sigma_guard_removed", TRAIN, "    assert want is None or abs(sig - float(want)) < 1e-9, f\"BFLC_SIG", "    assert True or abs(sig - float(want)) < 1e-9, f\"BFLC_SIG", ["frozen_set_and_checks_at_the_start"]),
    ("seed_guard_removed", TRAIN, "    assert m is None or int(m.group(1)) == a.seed, f\"--seed", "    assert True or int(m.group(1)) == a.seed, f\"--seed", ["frozen_set_and_checks_at_the_start"]),
    ("other_spec_resumed", TRAIN, "        assert json.load(open(p)) == want, f\"{p} holds another spec", "        assert True or json.load(open(p)) == want, f\"{p} holds another spec", ["fine_tune_records_resume_and_the_other_arms"]),
    ("no_resume", TRAIN, "ckpt_every=500, seed=a.seed, args=args,", "ckpt_every=500, resume=False, seed=a.seed, args=args,", ["fine_tune_records_resume_and_the_other_arms"]),
    ("spec_not_in_the_checkpoint", TRAIN, "init_run=run_init, narrow=model.narrow_spec, steps=steps", "init_run=run_init, narrow={}, steps=steps", ["fine_tune_records_resume_and_the_other_arms"]),
    ("calibration_copy_not_kept", TRAIN, "        shutil.copyfile(PP.rootpath(spec[\"calib\"]), run.path(\"calib.pt\"))\n        assert JN.sha256_file(run.path(\"calib.pt\")) == spec[\"calib_sha256\"]\n", "        pass\n", ["fine_tune_records_resume_and_the_other_arms"]),
]


def main():
    want = sys.argv[1:]
    res = {}
    for name, rel, old, new, tests in MUTANTS:
        if want and name not in want:
            continue
        p = os.path.join(ROOT, rel)
        src = open(p, encoding="utf-8").read()
        assert src.count(old) == 1, f"{name}: the text to mutate is found {src.count(old)} times"
        shutil.copyfile(p, p + ".orig_mut")
        t0 = time.time()
        try:
            open(p, "w", encoding="utf-8").write(src.replace(old, new))
            r = subprocess.run(["nice", "bash", "tools/env/_py14.sh", "grid/test_jnarrow.py", *tests], cwd=ROOT, capture_output=True, text=True)
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
