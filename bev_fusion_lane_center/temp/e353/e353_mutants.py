"""mutation check of test_jnarrow.py (scratch, one-time use, not promoted code): each mutant is applied to the real file, the named tests run, the file is restored; a mutant is KILLED when a test fails (the line shown is the first error line of the output).
Nothing may edit the mutated files while it runs. Run in WSL: python3 temp/e353/e353_mutants.py [names]   (2026-11-15: the 19 mutants of the build + the 29 of the critic pass 1)"""
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GRID, TRAIN, EVAL = "training/grid/jnarrow.py", "training/train/jnarrow_train.py", "training/train/jnarrow_eval.py"
FT = "fine_tune_records_resume_and_the_other_arms"
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
    ("narrowed_dump_through_the_stored_loader", EVAL, "    JEV.load_model = JN.load_model", "    pass", ["dump_wrapper_and_score_check", "timing_machinery_on_a_narrowed_toy"]),
    ("pyramid_left_trainable", TRAIN, "    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters())", "    ps = list(model.je.parameters()) + list(model.jmodel.jm.parameters()) + list(model.jmodel.pyr.parameters())", ["frozen_set_and_checks_at_the_start"]),
    ("node_head_left_trainable", TRAIN, "    for p in model.parameters():\n        p.requires_grad_(False)\n    for p in ps:", "    for p in model.parameters():\n        p.requires_grad_(False)\n    for p in model.nodes.parameters():\n        p.requires_grad_(True)\n    for p in ps:", ["frozen_set_and_checks_at_the_start"]),
    ("sigma_guard_removed", TRAIN, "    assert want is None or abs(sig - float(want)) < 1e-9, f\"BFLC_SIG", "    assert True or abs(sig - float(want)) < 1e-9, f\"BFLC_SIG", ["frozen_set_and_checks_at_the_start"]),
    ("seed_guard_removed", TRAIN, "    assert m is None or int(m.group(1)) == a.seed, f\"--seed", "    assert True or int(m.group(1)) == a.seed, f\"--seed", ["frozen_set_and_checks_at_the_start"]),
    ("other_spec_resumed", TRAIN, "        assert json.load(open(p)) == want, f\"{p} holds another spec", "        assert True or json.load(open(p)) == want, f\"{p} holds another spec", ["fine_tune_records_resume_and_the_other_arms"]),
    ("no_resume", TRAIN, "ckpt_every=500, seed=a.seed, args=args,", "ckpt_every=500, resume=False, seed=a.seed, args=args,", ["fine_tune_records_resume_and_the_other_arms"]),
    ("spec_not_in_the_checkpoint", TRAIN, "init_run=run_init, narrow=model.narrow_spec, steps=steps", "init_run=run_init, narrow={}, steps=steps", ["fine_tune_records_resume_and_the_other_arms"]),
    ("calibration_copy_not_kept", TRAIN, "        shutil.copyfile(PP.rootpath(spec[\"calib\"]), run.path(\"calib.pt\"))\n        assert JN.sha256_file(run.path(\"calib.pt\")) == spec[\"calib_sha256\"]\n", "        pass\n", ["fine_tune_records_resume_and_the_other_arms"]),
    # ---- critic pass 1 (2026-11-15): the rules the first 19 left unguarded
    ("e2e_flag_removed", TRAIN, "    model.e2e = True ", "    pass ", [FT]),
    ("maps_loss_weight_dropped", TRAIN, "        loss = loss + a.maps_w * lm\n", "        loss = loss\n", ["loss_is_the_closure_of_jends_train", FT]),
    ("geom_w_changed_in_the_loss", TRAIN, "cross_w=a.cross_w, driven_w=a.driven_w, geom_w=a.geom_w)", "cross_w=a.cross_w, driven_w=a.driven_w, geom_w=0.0)", ["loss_is_the_closure_of_jends_train"]),
    ("ppts_flag_off", TRAIN, "LD.QData.DEFAULT_PPTS = a.geom_w > 0", "LD.QData.DEFAULT_PPTS = False", [FT]),
    ("scales_flag_off", TRAIN, "LD.QData.DEFAULT_PSCALE = a.scales ", "LD.QData.DEFAULT_PSCALE = False ", [FT]),
    ("seed_zero_for_the_fit", TRAIN, "ckpt_every=500, seed=a.seed, args=args,", "ckpt_every=500, seed=0, args=args,", [FT]),
    ("shuffle_off", TRAIN, 'workers=a.workers, shuffle=True, map_mode="train"', 'workers=a.workers, shuffle=False, map_mode="train"', [FT]),
    ("maps_w_default_zero", TRAIN, "            setattr(a, k, args0[k])", "            setattr(a, k, 0.0 if k == 'maps_w' else args0[k])", [FT]),
    ("geom_w_default_zero", TRAIN, "            setattr(a, k, args0[k])", "            setattr(a, k, 0.0 if k == 'geom_w' else args0[k])", [FT]),
    ("calibration_sha_check_removed", GRID, '        assert sha256_file(PP.rootpath(spec["calib"])) == spec["calib_sha256"], "the calibration file is not the one the spec was resolved on"', "        pass", ["spec_round_trip_and_loaders"]),
    ("name_may_be_the_init_run", TRAIN, '    assert name != run_init, f"--name {name} is the init run: its folder is a stored run"', "    pass", ["run_folder_rule", FT]),
    ("folder_rule_removed", TRAIN, '        assert a0.get("init_run") == run_init and a0.get("seed") == seed, f"{folder} holds', '        assert True or a0.get("seed") == seed, f"{folder} holds', ["run_folder_rule"]),
    ("folder_rule_ignores_the_seed", TRAIN, 'a0.get("init_run") == run_init and a0.get("seed") == seed, f"{folder} holds', 'a0.get("init_run") == run_init, f"{folder} holds', ["run_folder_rule"]),
    ("folder_rule_ignores_the_init_run", TRAIN, 'a0.get("init_run") == run_init and a0.get("seed") == seed, f"{folder} holds', 'a0.get("seed") == seed, f"{folder} holds', ["run_folder_rule"]),
    ("folder_rule_not_called", TRAIN, "    check_name(a.name, run.dir, run_init, a.seed) ", "    pass ", [FT]),
    ("attach_without_geom_allowed", TRAIN, '    assert not a.attach or a.geom_w > 0, "--attach needs', '    assert True or a.geom_w > 0, "--attach needs', ["frozen_set_and_checks_at_the_start"]),
    ("macs_not_logged", TRAIN, '    run.log(f"measured GMACs per frame (FlopCounterMode, batch 1; init -> this run): {macs_line(macs0, macs1)}")', "    pass", [FT]),
    ("macs_of_the_run_not_measured", TRAIN, 'macs1 = JN.measured_macs(model) if model.narrow_spec["taps"] else macs0', "macs1 = macs0", [FT]),
    ("readers_with_stale_widths", GRID, 'def width(model, W, tensor):\n    return model.jmodel.pyr.c_in if tensor == "x" else PC.sub(model, producer_of(W, tensor)).out_channels',
     '_STALE = {}\n\n\ndef width(model, W, tensor):\n    return _STALE.setdefault((id(model), tensor), model.jmodel.pyr.c_in if tensor == "x" else PC.sub(model, producer_of(W, tensor)).out_channels)', ["two_edited_taps_of_one_reader"]),
    ("dump_of_a_stored_run_allowed", EVAL, '    assert a.unnarrowed_ok or written_by_jnarrow(a.run), f"', '    assert True or written_by_jnarrow(a.run), f"', ["dump_wrapper_and_score_check"]),
    ("unnarrowed_ok_ignored", EVAL, '    assert a.unnarrowed_ok or written_by_jnarrow(a.run), f"', '    assert written_by_jnarrow(a.run), f"', ["dump_wrapper_and_score_check"]),
    ("cp_control_not_a_jnarrow_run", EVAL, 'return "narrow" in args and "init_run" in args', 'return bool(args.get("narrow")) and "init_run" in args', ["dump_wrapper_and_score_check"]),
    ("paired_difference_sign_flipped", EVAL, "d = JT.paired(blocks[n], blocks[c])", "d = JT.paired(blocks[c], blocks[n])", ["paired_timing"]),
    ("paired_blocks_not_alternating", EVAL, "for k in JT.schedule(list(fns), r):", "for k in list(fns):", ["paired_timing"]),
    ("paired_pause_removed", EVAL, "    time.sleep(SETTLE_S)", "    pass", ["paired_timing"]),
    ("paired_busy_gpu_not_refused", EVAL, "    if not JT.gpu_idle(before) and not a.force:", "    if False:", ["paired_timing"]),
    ("paired_file_overwritten", EVAL, '    assert not os.path.exists(out), f"{out} exists: another --tag (a measurement is never overwritten)"', "    pass", ["paired_timing"]),
    ("paired_dispatch_removed", EVAL, '    if argv[:1] == ["paired"]:\n        return paired_cmd(argv[1:])\n', "", ["paired_timing"]),
    ("paired_quiet_verdict_ignored", EVAL, 'quiet, line = JT.quiet_verdict(host["load_before"], host["load_after"], host["gpu_before"], host["gpu_after"])', 'quiet, line = True, "QUIET"', ["paired_timing"]),
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
        mutated = src.replace(old, new)
        compile(mutated, p, "exec")                                                           # a mutant that does not compile would be "killed" for the wrong reason
        shutil.copyfile(p, p + ".orig_mut")
        t0 = time.time()
        try:
            open(p, "w", encoding="utf-8").write(mutated)
            r = subprocess.run(["nice", "bash", "tools/env/_py14.sh", "grid/test_jnarrow.py", *tests], cwd=ROOT, capture_output=True, text=True)
        finally:
            shutil.move(p + ".orig_mut", p)
        killed = r.returncode != 0
        lines = [x for x in (r.stdout + r.stderr).strip().split("\n") if x]
        tail = ([x for x in lines if "Error" in x or "assert" in x.lower()] or lines)[-1][:160]
        res[name] = killed
        print(f"{'KILLED  ' if killed else 'SURVIVED'} {name} ({time.time() - t0:.0f} s): {tail}", flush=True)
        assert open(p, encoding="utf-8").read() == src, "restore failed"
    print("survivors:", [n for n, k in res.items() if not k])


if __name__ == "__main__":
    main()
