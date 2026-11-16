p = "temp/e353/e352_mutants.py"
s = open(p, encoding="utf-8").read()
a = s.index("MUTANTS = [")
b = s.index("def main():")
new = '''MUTANTS = [
    ("edit_not_applied_to_B", "    shim, wb = PP.ablate(Shim(qg), edits, taps)", "    shim, wb = PP.ablate(Shim(qg), [], taps)", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
    ("edit_not_applied_to_the_junction_model", "    mj, wj = PP.ablate(jm, edits, taps)", "    mj, wj = PP.ablate(jm, [], taps)", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
    ("random_is_ranked", "PP.edits_for(u, f, calib, how, rng)", "PP.edits_for(u, f, calib, \\"rank\\", None)", ["points_ladder_counts_and_reference_channels"]),
    ("random_seed_without_the_unit", "|{draw}|{unit}\\".encode()", "|{draw}\\".encode()", ["points_ladder_counts_and_reference_channels"]),
    ("random_seed_without_the_variant", "e352|{'groups' if groups else 'channels'}|", "e352|channels|", ["groups_variant_references"]),
    ("random_seed_sum_of_ord", "zlib.crc32(f\\"e352|{'groups' if groups else 'channels'}|{draw}|{unit}\\".encode())", "1000 * draw + sum(map(ord, unit))", ["points_ladder_counts_and_reference_channels"]),
    ("group_ranking_reversed", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), stable=True)", "torch.argsort(calib.s[tap].double().view(G, gs).sum(1), descending=True, stable=True)", ["groups_variant_references"]),
    ("group_means_misaligned", "val=calib.mean[tap][idx]))\n    return out", "val=calib.mean[tap][:len(idx)]))\n    return out", ["groups_variant_references"]),
    ("random_groups_reversed", "rng.permutation(G)[:n].tolist()", "rng.permutation(G)[::-1][:n].tolist()", ["groups_variant_references"]),
    ("premise_not_asserted", "    assert len(keys) == 57 and all(torch.equal(sb[k], sj[\\"jmodel.\\" + k]) for k in keys), \\"B's pyramid", "    assert True or all(torch.equal(sb[k], sj[\\"jmodel.\\" + k]) for k in keys), \\"B's pyramid", ["pyramid_premise_and_shim"]),
    ("seed_one_loads_seed_zero_calibration", "return PP.rootpath(f\\"temp/e345/e324_s_s{seed}/calib.pt\\")", "return PP.rootpath(\\"temp/e345/e324_s_s0/calib.pt\\")", ["seed_one_uses_its_own_calibration"]),
    ("overwrite_allowed", "    if os.path.exists(os.path.join(folder, \\"point.json\\")):\n        raise SystemExit(f\\"{folder} is complete already: a point folder is never overwritten\\")\n    os.makedirs", "    os.makedirs", ["point_folder_files_hashes_and_refusals"]),
    ("link_to_B_checkpoint", "os.symlink(os.path.abspath(os.path.join(ROOT, \\"runs\\", j, \\"model.pt\\")), link)", "os.symlink(os.path.abspath(os.path.join(ROOT, \\"runs\\", b, \\"model.pt\\")), link)", ["point_folder_files_hashes_and_refusals"]),
    ("reversed_predictions_in_the_dump", "RO.dump_full(res[\\"preds\\"], res[\\"rows\\"], pb + \\".tmp\\")", "RO.dump_full(res[\\"preds\\"][::-1], res[\\"rows\\"], pb + \\".tmp\\")", ["point_folder_files_hashes_and_refusals"]),
    ("partial_flag_always_false", "partial=len(res[\\"frames\\"]) != res[\\"total\\"]", "partial=False", ["point_folder_files_hashes_and_refusals"]),
    ("main_does_not_check_the_partial_dump", "        check_out(a.max, a.out)\n", "", ["partial_dump_cannot_land_in_runs"]),
    ("check_out_allows_runs", "    if os.path.realpath(PP.rootpath(out)).startswith(os.path.realpath(os.path.join(ROOT, \\"runs\\")) + os.sep):", "    if False:", ["partial_dump_cannot_land_in_runs"]),
    ("score_line_without_base", " --base $R/runs/e273_p_s{seed}/model.pt", "", ["scoring_lines_pass_the_stored_checks"]),
    ("score_line_wrong_tag", "--spec close:1.5@0.5 --tag {tag}\\"", "--spec close:1.5@0.5 --tag {tag}x\\"", ["scoring_lines_pass_the_stored_checks"]),
    ("run_id_without_variant", "return f\\"E352.s0{variant(name, groups)}_{name}", "return f\\"E352.s0_{name}", ["names_ids_folders_and_lines"]),
    ("comparison_blind_to_nodes", "        r[\\"nodes_only_in_ref\\"] += len(um_a)\n        r[\\"nodes_only_in_new\\"] += len(um_b)\n", "", ["dump_comparison_catches_a_changed_dump"]),
    ("comparison_ignores_links", "        r[\\"links_only_in_ref\\"] += len(one_a)\n        r[\\"links_only_in_new\\"] += len(one_b)\n", "", ["dump_comparison_catches_a_changed_dump"]),
    ("comparison_ignores_sections", "        r[\\"sections_differ\\"] += int((mapa[sa[both]] != sb[both]).sum() + ((sa >= 0) != (sb >= 0)).sum())\n", "", ["dump_comparison_catches_a_changed_dump"]),
    ("comparison_skips_logits", "            for k in (\\"logit\\", \\"scale\\"):\n                if isinstance(a.get(k), np.ndarray):\n                    put(k,", "            for k in ():\n                if isinstance(a.get(k), np.ndarray):\n                    put(k,", ["empty_set_reproduces_the_stored_dumps"]),
    ("dump_with_wrong_frame_order", "    assert [f[\\"row\\"] for f in frames] == rows[:n], \\"the junction frames are not in the order of B's loader\\"", "    frames = frames[::-1]", ["edited_pass_changes_both_models_and_the_random_draw_differs"]),
    ("gate_topo_tolerance_loose", "GATE_TOPO_F1 = 2e-4 ", "GATE_TOPO_F1 = 2e-2 ", ["gate_tolerances_and_real_cards"]),
    ("gate_ignores_the_head", "            for which in (\\"base\\", \\"head\\"):\n            for key in KEYS:", "            for which in (\\"base\\",):\n            for key in KEYS:", ["gate_tolerances_and_real_cards"]),
    ("gate_relative_bound_dropped", "    return GATE_TOPO_F1 if key == \\"topo_F1\\" else max(GATE_ABS, GATE_REL * abs(stored))", "    return GATE_TOPO_F1 if key == \\"topo_F1\\" else GATE_ABS", ["gate_tolerances_and_real_cards"]),
    ("table_ranked_minus_random_sign", "gap = r[0] - m[0]", "gap = m[0] - r[0]", ["table_known_answers_and_the_rule_lines"]),
    ("table_kill_rule_inverted", "and (m[0] - n[0]) >= (r[0] - n[0])", "and (m[0] - n[0]) < (r[0] - n[0])", ["table_known_answers_and_the_rule_lines"]),
    ("table_loss_against_the_wrong_null", "f\\"{x - b:+.4f}\\" if delta else \\"-\\"", "f\\"{x - n[0]:+.4f}\\" if delta else \\"-\\"", ["table_known_answers_and_the_rule_lines"]),
]


'''
s = s[:a] + new + s[b:]
open(p, "w", encoding="utf-8").write(s)
