"""one-off: do the new tests fail on the mutants the critic named? Each mutant edits the SOURCE of one function in memory (the files are not touched), runs the test that should catch it and restores the function."""
import inspect
import os
import sys
import textwrap
import traceback

sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation", "tests"))
import test_jprune_probe as T  # noqa: E402
import jprune_probe as PP  # noqa: E402
import jprune_table as PT  # noqa: E402

MUTANTS = [
    ("dup_frac sign", PT, PT.RunData, "full_loss_by_readout", 'sign = -1.0 if key == "dup_frac" else 1.0', "sign = 1.0", T.test_full_loss_on_stored_cards),
    ("worse read-out = min", PT, PT.RunData, "full_loss", "max(ls)", "min(ls)", T.test_full_loss_on_stored_cards),
    ("dpos ignores the attached end", PT, PT.RunData, "delta", 'dpos=max(a["pos"] - n["pos"], a["apos"] - n["apos"])', 'dpos=a["pos"] - n["pos"]', T.test_sweep_position_columns_and_dpos),
    ("dpos sign", PT, PT.RunData, "delta", 'dpos=max(a["pos"] - n["pos"], a["apos"] - n["apos"])', 'dpos=max(n["pos"] - a["pos"], n["apos"] - a["apos"])', T.test_sweep_position_columns_and_dpos),
    ("sweep position column (angle instead of position)", PT, PT.RunData, "sweep", "pos=float(k[1] / max(k[0], 1))", "pos=float(k[2] / max(k[0], 1))", T.test_sweep_position_columns_and_dpos),
    ("sweep attached column", PT, PT.RunData, "sweep", "apos=float(k[3] / max(k[0], 1))", "apos=float(k[4] / max(k[0], 1))", T.test_sweep_position_columns_and_dpos),
    ("sweep half mask dropped", PT, PT.RunData, "sweep", "m = self.rec[\"passes\"][key], self.mask(half)", "m = self.rec[\"passes\"][key], self.mask(\"all\")", T.test_sweep_position_columns_and_dpos),
    ("klass: best seed instead of worst", PT, PT.Model, "klass", "max(", "min(", T.test_worse_seed_in_klass_and_free_fraction),
    ("free_fraction: any seed instead of every", PT, PT.Model, "free_fraction", "if all(", "if any(", T.test_worse_seed_in_klass_and_free_fraction),
    ("run_pass: selection and judgement curves swapped", PP, PP, "run_pass", '("sel", half == 0), ("jud", half == 1)', '("sel", half == 1), ("jud", half == 0)', T.test_run_pass_pair_curves_by_half),
    ("plan: v threshold", PT, PT, "plan", "<= FULL_LIMITS[0]]", "<= FULL_LIMITS[1]]", T.test_plan_top_savers_are_heads_and_v_composition),
    ("plan: handle-scale fraction from the sweep", PT, PT, "plan", "if u.name in handle else", "if False else", T.test_plan_top_savers_are_heads_and_v_composition),
    ("plan: top savers include the pyramid", PT, PT, "plan", 'for u in model.units if u.group in ("maps", "pair") and model.free_fraction(u.name, [r0]) > 0]', "for u in model.units if model.free_fraction(u.name, [r0]) > 0]", T.test_plan_top_savers_are_heads_and_v_composition),
    ("plan: v includes the pyramid", PT, PT, "plan", 'if u.group in ("maps", "pair"):\n                f =', "if True:\n                f =", T.test_plan_top_savers_are_heads_and_v_composition),
    ("greedy: by loss instead of loss per GMAC", PT, PT.Model, "greedy", 'else i["dloss"] / max(i["dmacs"], 1e-12)', 'else i["dloss"]', T.test_greedy_orders_by_loss_per_gmac),
    ("control: ranked vs random reversed", PT, PT, "control_report", "rank += a < rd", "rank += a > rd", T.test_control_report_shares_differ),
    ("control: reverse vs ranked reversed", PT, PT, "control_report", "rev += rv > a", "rev += rv < a", T.test_control_report_shares_differ),
    ("control: random draws by min", PT, PT, "control_report", "float(np.mean([run.delta", "float(np.min([run.delta", T.test_control_report_shares_differ),
    ("pair_candidates keypoint threshold 0.5", PP, PP, "pair_candidates", "kp_thr=THR", "kp_thr=KP_THR", T.test_pair_curve_and_candidates_known_answers),
    ("kp_error keypoint threshold 0.1", PP, PP, "kp_error", "kp_thr=KP_THR", "kp_thr=THR", T.test_pair_curve_and_candidates_known_answers),
    ("sizes: no floor", PT, PT, "sizes", "KNEE_FLOOR", "1e-9", T.test_classes_greedy_prefixes_sizes),
    ("spearman without tie handling", PT, PT, "spearman", "rankdata(a), rankdata(b)", "np.argsort(np.argsort(a)), np.argsort(np.argsort(b))", T.test_control_report_shares_differ),
    ("score_point: every read-out for every point", PT, PT, "score_point", "readouts = tuple(READOUTS[family]) if pid in (0, NULL2) else JUDGED[family]", "readouts = tuple(READOUTS[family])", T.test_score_point_calls_the_stored_scorer),
    ("loss_upper: smaller upper end", PT, PT.RunData, "loss_upper", "max(x[", "min(x[", T.test_boot_on_stored_cards),
    ("boot: loss sign not negated", PT, PT, "boot_point", "dict(point=-pt, mean=-mn, lo=-hi, hi=-lo)", "dict(point=pt, mean=mn, lo=lo, hi=hi)", T.test_boot_on_stored_cards),
    ("klass: no interval guard", PT, PT.Model, "klass", 'return ("FREE" if max(ups) <= FREE_UPPER else "CHEAP"), "full + interval", loss', 'return "FREE", "full + interval", loss', T.test_worse_seed_in_klass_and_free_fraction),
    ("dump_point: existence of the pickle instead of point.json", PP, PP.Sweep, "dump_point", 'if os.path.exists(os.path.join(d, "point.json")):', 'if os.path.exists(os.path.join(d, "ends_val.pkl")):', T.test_sweep_passes_and_point_folder),
    ("wait_for_gpu: goes on into a full GPU", PP, PP, "wait_for_gpu", "sys.exit(75)", "pass", T.test_gpu_gate_wrapper_and_deterministic_setting),
    ("resolve_deterministic: mixed settings accepted", PP, PP, "resolve_deterministic", 'if requested and not was and rec.get("passes"):', "if False:", T.test_gpu_gate_wrapper_and_deterministic_setting),
]
only = sys.argv[1:]
res = []
for name, mod, owner, fn, old, new, test in MUTANTS:
    if only and not any(o in name for o in only):
        continue
    target = getattr(owner, fn)
    src = textwrap.dedent(inspect.getsource(target))
    assert old in src, (name, "pattern not found")
    ns = {}
    exec(compile(src.replace(old, new), "<mutant>", "exec"), mod.__dict__, ns)
    setattr(owner, fn, ns[fn])
    try:
        test()
        res.append((name, "SURVIVED"))
    except BaseException as ex:                                                          # the mutant is caught by an assertion (or by any error it causes)
        res.append((name, "killed (" + type(ex).__name__ + ")"))
    finally:
        setattr(owner, fn, target)
for r in res:
    print(r[1].ljust(28), r[0])
print("survivors:", [n for n, s in res if s == "SURVIVED"])
