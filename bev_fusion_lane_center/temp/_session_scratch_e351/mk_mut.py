"""scratch: mutants of evaluation/jgraph.py and evaluation/jtiming.py (one exact replacement each) for the mutation evidence of E351 (round 2); run_mut.sh runs the selftest of each (it must FAIL)"""
import os
import shutil

R = os.environ["BFLC_ROOT"]
OUT = os.path.join(R, "temp", "_session_scratch_e351", "mut")
J, T = "jgraph.py", "jtiming.py"
M = [
    # round 2 (critic F1 - F10), first
    ("F1a_gpu_ignored_in_quiet", T, " and gpu_idle(gpu_before) and gpu_idle(gpu_after)", ""),
    ("F1b_gpu_threshold", T, "state[\"util_max\"] <= 5.0", "state[\"util_max\"] <= 50.0"),
    ("F1c_not_quiet_ignored", T, "    if not quiet:\n        return \"INCONCLUSIVE\"", "    if False:\n        return \"INCONCLUSIVE\""),
    ("F2a_precheck_off", T, "    if cmp.bad:\n        sys.exit(\"STOP: the graph forward", "    if False:\n        sys.exit(\"STOP: the graph forward"),
    ("F2b_precheck_self_compare", T, "cmp.add(JG.eager_forward(model, x), gm.forward(x))", "cmp.add(JG.eager_forward(model, x), JG.eager_forward(model, x))"),
    ("F3a_partial_has_verdict", J, "    if graph[\"frames\"] != total or graph[\"tensors\"] == 0 or null[\"tensors\"] == 0:", "    if False:"),
    ("F3b_frames_not_counted", J, "    if graph[\"frames\"] != total or graph[\"tensors\"] == 0 or null[\"tensors\"] == 0:", "    if graph[\"tensors\"] == 0 or null[\"tensors\"] == 0:"),
    ("F3c_tensors_not_counted", J, "    if graph[\"frames\"] != total or graph[\"tensors\"] == 0 or null[\"tensors\"] == 0:", "    if graph[\"frames\"] != total:"),
    ("F3d_max_without_tag", J, "    assert not a.max or a.tag or runs_dir, ", "    assert True or not a.max or a.tag or runs_dir, "),
    ("F3e_dump_max_into_runs", J, "    assert not (a.dump and a.max) or runs_dir, ", "    assert True or not (a.dump and a.max) or runs_dir, "),
    ("F3f_result_overwritable", J, "    assert not os.path.exists(out), f\"{out} exists", "    assert True or not os.path.exists(out), f\"{out} exists"),
    ("F4a_expected_set_ignored", J, "        if found != sorted(expected):\n            diffs.append(", "        if False:\n            diffs.append("),
    ("F4b_expected_only_snap", J, "for fl in score_flags(attach))\n", "for fl in score_flags(False))\n"),
    ("F5a_missing_dir_ok", J, "            ok = False\n            continue", "            continue"),
    ("F5b_runs_dir_note_dropped", J, "    if runs_dir:\n        lines.append(\"note:", "    if False:\n        lines.append(\"note:"),
    ("F6_bits_ignored", J, "    if torch.equal(bits(a), bits(b)):\n        return None", "    if torch.equal(a, b):\n        return None"),
    ("F7_training_allowed", J, "        assert not model.training, ", "        assert True or not model.training, "),
    ("F8a_rel_absolute", J, "    return os.path.relpath(path, ROOT).replace(os.sep, \"/\") if os.path.abspath(path).startswith(ROOT + os.sep) else path", "    return path"),
    ("F8b_resolve_dir_plain", J, "    return os.path.join(ROOT, path) if path else None", "    return path or None"),
    ("F10_empty_dump_crash", J, "    if ns.attach and frames and \"att_pos\" not in frames[0]:", "    if ns.attach and \"att_pos\" not in frames[0]:"),
    # the rest (unchanged logic of round 1, re-run on the final code)
    ("g01_dtype_ignored", J, "    if a.device != b.device or a.dtype != b.dtype or a.shape != b.shape:", "    if a.device != b.device or a.shape != b.shape:"),
    ("g02_leaf_not_compared", J, "                if not bool(a == b):", "                if False:"),
    ("g04_no_copy_into_static", J, "        self.static_x.copy_(x)\n        self.runner.replay()", "        self.runner.replay()"),
    ("g05_no_clone", J, "        return clone_tree((P, out, oe))", "        return (P, out, oe)"),
    ("g06_dump_overwrite", J, "    os.makedirs(dst)", "    os.makedirs(dst, exist_ok=True)"),
    ("g07_seed_unchecked", J, "    if not (args.get(\"seed\", seed) == seed and (m is None or int(m.group(1)) == seed)):", "    if False:"),
    ("g08_counts_ignored", J, "        if not (os.path.isfile(ka) and os.path.isfile(kb) and same_tree(pickle.load(open(ka, \"rb\")), pickle.load(open(kb, \"rb\")))):", "        if False:"),
    ("g09_forward_hash_off", J, "        assert forward_sha(model) == FORWARD_SHA, ", "        assert True or forward_sha(model) == FORWARD_SHA, "),
    ("g10_pool_not_overwritten", J, "            copy_into(self.out, self.fn())", "            self.out = self.fn()"),
    ("g11_nodes_dropped", J, "nodes=nd if self.model.je.attach is not None else None)", "nodes=None)"),
    ("g12_preflight_off", J, "        assert not os.path.exists(dump_dir(run, arm, runs_dir)), f\"", "        assert True or not os.path.exists(dump_dir(run, arm, runs_dir)), f\""),
    ("g14_missing_key_ignored", J, "                self._flag(k, \"missing\" if k not in t else \"extra\")\n                continue", "                continue"),
    ("g15_tensor_count", J, "                self.tensors += 1", "                self.tensors += 0"),
    ("g16_pair_head_without_autocast", J, "        with inference(x.device.type, self.dtype):\n            oe = ", "        with inference(x.device.type, None):\n            oe = "),
    ("t01_adopt_threshold", T, "    if mean <= -need:", "    if mean <= need:"),
    ("t02_quiet_limit_inclusive", T, "load_before < limit and load_after < limit", "load_before <= limit and load_after <= limit"),
]
shutil.rmtree(OUT, ignore_errors=True)
for name, f, old, new in M:
    src = open(os.path.join(R, "evaluation", f), encoding="utf-8", newline="").read()
    assert src.count(old) == 1, (name, src.count(old))
    d = os.path.join(OUT, name)
    os.makedirs(d)
    open(os.path.join(d, f), "w", encoding="utf-8", newline="").write(src.replace(old, new))
print(len(M), "mutants in", OUT)
open(os.path.join(OUT, "list.txt"), "w").write("\n".join(f"{n} {f}" for n, f, _, _ in M) + "\n")
