R = "/mnt/c/Auto-research/bev_fusion_lane_center/"
blk = open(R + "temp/e345/newblock.py", encoding="utf-8").read()
for old, new in (("(0.012 if fr == 0.5 else 0.001)", "(0.008 if fr == 0.5 else 0.001)"), ("# seed 1 loses 0.012 at f = 0.5", "# seed 1 loses 0.008 at f = 0.5"), ("abs(mod.klass(pid)[2] - 0.012) < 1e-9", "abs(mod.klass(pid)[2] - 0.008) < 1e-9"),
                 ('''    full[scale[0]] = 0.001
    full[scale[1]] = 0.006
''', '''    full[scale[0]] = full[scale[1]] = 0.006
'''), ("PT.spearman([1, 2, 3], [3, 2, 1]) == -1.0", "abs(PT.spearman([1, 2, 3], [3, 2, 1]) + 1.0) < 1e-12")):
    assert blk.count(old) == 1, old
    blk = blk.replace(old, new)
p = R + "evaluation/tests/test_jprune_probe.py"
s = open(p, encoding="utf-8").read()
a = s.index("def test_score_point_calls_the_stored_scorer():")
b = s.index("TESTS = [")
rest = s[b:]
rest = rest[:rest.index("]\n") + 2]
tail = s[b + len(rest):]
new_tests = '''TESTS = [test_unit_table_both_models, test_paint_columns_known_answer, test_pair_curve_and_candidates_known_answers, test_halves_are_the_cards, test_sweep_position_columns_and_dpos, test_worse_seed_in_klass_and_free_fraction,
         test_classes_greedy_prefixes_sizes, test_greedy_orders_by_loss_per_gmac, test_plan_top_savers_are_heads_and_v_composition, test_plan_ids_second_null_and_sets_report, test_control_report_shares_differ, test_full_loss_on_stored_cards,
         test_boot_on_stored_cards, test_score_point_calls_the_stored_scorer, test_gpu_gate_wrapper_and_deterministic_setting, test_dump_frames_equals_jends_eval_dump, test_frame_stats_and_attached_fields, test_run_pass_pair_curves_by_half,
         test_calibration_against_numpy_reference, test_appliers_against_independent_references, test_plumbing_controls, test_dead_hook_assert_catches, test_saved_cost_against_narrowed_copies, test_sweep_passes_and_point_folder]
'''
s = s[:a] + blk + new_tests + tail
open(p, "w", encoding="utf-8").write(s)
print("ok", len(s.splitlines()))
