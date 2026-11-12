"""one-off (E275): patch evaluation/qgraph_run_eval.py for the critic's findings (topo_guard flag, stricter verdict275, repro gate, tied pairs, explicit --exp, title with the read-out)"""
p = "evaluation/qgraph_run_eval.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, cnt=1):
    global s
    assert s.count(old) == cnt, (s.count(old), old[:90])
    s = s.replace(old, new)


# --- verdict274: topo_guard flag
rep('def verdict274(var, ctl, exp_var="e274", exp_ctl="e273", metric="topo_F1", band=0.02, root=None, verbose=True):',
    'def verdict274(var, ctl, exp_var="e274", exp_ctl="e273", metric="topo_F1", band=0.02, root=None, verbose=True, topo_guard=True):')
rep('    if metric != "topo_F1":\n        guards["topo_F1_not_below_control_by_0_01"] = all(g >= -0.01 for g in topo_gain)',
    '    if metric != "topo_F1" and topo_guard:                                                    # the E274 card\'s guard for arm p against r; the E275 card has none (verdict275 passes False)\n'
    '        guards["topo_F1_not_below_control_by_0_01"] = all(g >= -0.01 for g in topo_gain)')

# --- verdict275: no topo guard, checks of what it judges
start = s.index("def verdict275(")
end = s.index("def main():")
new_v = '''def verdict275(var, ctl, exp_var="e275", exp_ctl="e273", root=None, lanes_tol=0.03, verbose=True, spec="close1.5_t0.5", frames=873):
    """the rule of the E275 card (section 0) for the close-node suppression (variant `var`: the cards of the step, control `ctl`: the same models read out as before; both lists in the same seed
    order). Two calls of verdict274 (topo_F1, band 0.02; junction_F1, band 0.03: each with the assignment guards and the F1@1.0 guard of that function, WITHOUT the topo guard of the E274 card) and
    one more guard: lanes_found not below the control by more than `lanes_tol` on any seed. KEEP: both headlines up on both seeds and every guard held. CUT: either headline down on a seed or any
    guard broken. TIE: both inside their bands and the guards held. Everything else: INCONCLUSIVE (goes to Mirko: there is no further seed without retraining).
    What it judges is checked first: two seeds (0 and 1) in the same order, `frames` frames in every card, the variant cards carry the read-out `spec`, the control cards none (or plain_t0.5)."""
    root = root or ROOT
    load = lambda exp, r: json.load(open(os.path.join(root, "runs", exp, r, "card.json")))  # noqa: E731
    cv, cc = [load(exp_var, r) for r in var], [load(exp_ctl, r) for r in ctl]
    assert [c["seed"] for c in cv] == [c["seed"] for c in cc] == [0, 1], f"the rule needs seeds 0 and 1 in this order, got {[c['seed'] for c in cv]} against {[c['seed'] for c in cc]}"
    assert all(c["frames"] == frames for c in cv + cc), f"every card must hold {frames} frames: {[c['frames'] for c in cv + cc]}"
    assert all(c.get("readout", {}).get("spec") == spec for c in cv), f"the variant cards must carry the read-out {spec}: {[c.get('readout', {}).get('spec') for c in cv]}"
    assert all(c.get("readout", {}).get("spec", "plain_t0.5") == "plain_t0.5" for c in cc), "the control cards must be read out as before (no read-out step)"
    t = verdict274(var, ctl, exp_var, exp_ctl, "topo_F1", 0.02, root, verbose=False, topo_guard=False)
    j = verdict274(var, ctl, exp_var, exp_ctl, "junction_F1", 0.03, root, verbose=False, topo_guard=False)
    lanes = [a["p"]["graph"]["lanes_found"] - b["p"]["graph"]["lanes_found"] for a, b in zip(cv, cc)]
    guards = dict(t["guards"])
    guards["lanes_found_not_below_control_by_%g" % lanes_tol] = all(x >= -lanes_tol for x in lanes)
    sides = (t["side"], j["side"])
    if "down" in sides or not all(guards.values()):
        v = "CUT"
    elif sides == ("up", "up"):
        v = "KEEP"
    elif sides == ("inside", "inside"):
        v = "TIE"
    else:
        v = "INCONCLUSIVE (the two headlines or the seeds disagree; goes to Mirko)"
    out = dict(verdict=v, topo_F1=dict(side=t["side"], gain_per_seed=t["gain_per_seed"]), junction_F1=dict(side=j["side"], gain_per_seed=j["gain_per_seed"]), lanes_found_gain_per_seed=lanes,
               F1_1_gain_per_seed=t["F1_1_gain_per_seed"], guards=guards, variant_numbers=t["variant_numbers"], variant=var, control=ctl,
               note="by hand: the written rule of the E275 card (section 0); in-sample on VAL; density and system pass are reported, not judged")
    if verbose:
        print(json.dumps(out, indent=1))
    return out


'''
s = s[:start] + new_v + s[end:]

# --- score(): repro gate, tied pairs
rep('def score(run, seed, n=0, device="cuda", exp="e273", readouts=None, curve=(), dump=None, max_drop=0.20):',
    'def score(run, seed, n=0, device="cuda", exp="e273", readouts=None, curve=(), dump=None, max_drop=0.20, repro=None, repro_tol=0.005):')
rep('    nodes delivered at its threshold stops the scoring before any card is written."""',
    "    nodes delivered at its threshold stops the scoring before any card is written. repro: path of the E274 sweep json of this model; the close:1.5@0.5 read-out must reproduce its entry\n"
    "    'pos1.5_e0.5' (topo F1, F1@1.0, junction F1, lanes found, node P / R) within repro_tol, else the scoring stops after the exact-map pass, before the control and before any card is written.\"\"\"")
rep("nodes_after=after, dropped_frac=1.0 - after / max(before, 1), ms_per_frame=RO.timed(preds, s))",
    "nodes_after=after, dropped_frac=1.0 - after / max(before, 1), tied_pairs_within_radius=RO.tied_pairs(preds, s), ms_per_frame=RO.timed(preds, s))")
rep("            res[s][mode], recs[s][mode] = out, rec\n        del preds",
    '            res[s][mode], recs[s][mode] = out, rec\n'
    '            if mode == "exact" and repro and s == RO.Spec("close", 1.5, 0.5):                          # layer 1: the named step must reproduce the scratch sweep of E274\n'
    '                ref = json.load(open(repro))["pos1.5_e0.5"]\n'
    '                mine = dict(topo_F1=out["graph"]["topo_F1"], F1_1=out["graph"]["F1@1.0"], junction_F1=out["graph"]["junction_F1"], lanes_found=out["graph"]["lanes_found"],\n'
    '                            node_P=out["nodes"]["node_P"], node_R=out["nodes"]["node_R"])\n'
    '                dif = {k: abs(mine[k] - ref[k]) for k in mine}\n'
    '                print(f"{run} close1.5_t0.5 against the E274 sweep: max abs difference {max(dif.values()):.2e} ({repro})", flush=True)\n'
    '                if max(dif.values()) > repro_tol:\n'
    '                    raise SystemExit(f"KILL CRITERION: the step does not reproduce the E274 sweep (tolerance {repro_tol}): {dif}: nothing written, search the code")\n'
    '                info[s]["reproduces_e274_sweep_max_abs_diff"] = max(dif.values())\n'
    '        del preds')

# --- write_md title with the read-out
rep('''    L = [f"# {c['run']} (seed {c['seed']}), VAL {c['frames']} frames, {c['date']}", "",''',
    '''    L = [f"# {c['run']}{' [' + c['readout']['spec'] + ']' if 'readout' in c else ''} (seed {c['seed']}), VAL {c['frames']} frames, {c['date']}", "",''')

# --- CLI
rep('    s.add_argument("--exp", default="e273", help="the experiment folder of the scoring card: runs/<exp>/<run>/")',
    '    s.add_argument("--exp", default=None, help="the experiment folder of the scoring card: runs/<exp>/<run>/ (default e273; required with --readouts, e.g. e275)")')
rep('    s.add_argument("--max-drop", type=float, default=0.20, help="E275 kill criterion: stop when a close-node spec drops more than this share of the delivered nodes")',
    '    s.add_argument("--max-drop", type=float, default=0.20, help="E275 kill criterion: stop when a close-node spec (threshold >= 0.5) drops more than this share of the delivered nodes")\n'
    '    s.add_argument("--repro", default=None, help="E275: the E274 sweep json of this model (temp/e274/sweep_<run>.json): close:1.5@0.5 must reproduce its pos1.5_e0.5 entry within 0.005")')
rep('    if a.cmd == "score":\n        dump =',
    '    if a.cmd == "score":\n        if a.readouts and not a.exp:\n            ap.error("--readouts needs --exp (e.g. e275): the default folder of the E273 cards must not receive read-out cards")\n        a.exp = a.exp or "e273"\n        dump =')
rep("dump=dump, max_drop=a.max_drop)", "dump=dump, max_drop=a.max_drop, repro=a.repro)")
open(p, "w", encoding="utf-8").write(s)
print("patched")
