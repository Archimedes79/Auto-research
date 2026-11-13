p = "evaluation/jmaneuver.py"
s = open(p, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, count)


# docstring: shell-safe variants, 0.5 m spacing, denominators
sub("    wsl bash tools/env/_py14.sh jmaneuver.py score --run e308_t_s0 --seed 0 [--variants plain p>=0.5 p>=0.1 oracle oracle+fit] [--rows]     CPU, VAL 873, from runs/<run>/ends_val.pkl (`jends_eval.py dump`) and B's full dump of the seed",
    "    wsl bash tools/env/_py14.sh jmaneuver.py score --run e308_t_s0 --seed 0 [--variants plain p0.5 p0.1 oracle oracle+fit] [--rows] [--chord] [--max N]     CPU, VAL 873, from runs/<run>/ends_val.pkl (`jends_eval.py dump`) and B's full dump of the seed")
sub("points of its GT run (every 1 m, inside the 100 m box) have", "points of its GT run (the GT points, spacing 0.5 m, inside the 100 m box; at least 3 points) have")
sub("Variants of the delivered graph (all on B's graph + the hard close-node step of the E275 spec): `plain` (no connectors), `p>=<t>` (the connectors of the run's pair logits at probability t, with the run's handle scales when its dump has them)",
    "Variants of the delivered graph (all on B's graph + the hard close-node step of the E275 spec): `plain` (no connectors), `p<t>` (written `p0.5`, shell-safe; `p>=0.5` is accepted when quoted: the connectors of the run's pair logits at probability t, with the run's handle scales when its dump has them, `--chord`: always the chord tangent)")
sub("`oracle` (a perfect pair classifier on the\nrun's keypoints: logit +20 for a candidate whose two keypoints are matched, strict one-to-one, to the two ends of a GT pair, -20 for every other; chord-tangent curves)",
    "`oracle` (a perfect pair classifier on the\nrun's keypoints (every `valid` one, like the `p<t>` variants): logit +20 for a candidate whose two keypoints are matched, strict one-to-one, to the two ends of a GT pair, -20 for every other; chord-tangent curves)")
sub("`oracle+fit` (the same with the handle scales fitted to the GT connector, `qjunction.fit_pair_scales`).", "`oracle+fit` (the same with the handle scales fitted to the GT connector, `qjunction.fit_pair_scales`; a pair whose fit is NaN, a GT chord under 1 m, keeps the chord curve).")
sub("With `--rows` the json also carries one row per GT maneuver (ends found by the keypoints, length, turn, range, realised per variant) for the decompositions of the E310 card.",
    "With `--rows` the json also carries one row per GT maneuver (ends found by a keypoint with p >= 0.5 within 4 m / 30 degrees, length, turn, range, realised per variant) for the decompositions of the E310 card. The maneuver denominator (826 frames, 10991 maneuvers on VAL) is smaller than the 11083 GT pairs of the pair numbers: the box and the 3-point rule drop 0.8 %. `--max N`: the first N frames only (a smoke test of the pipeline).")
# realised_share: all neighbours within TOL, not the 6 nearest
old_a = s.index("def realised_share(tree, tang, run_pts, run_tan):")
old_b = s.index("def frame_counts(")
s = s[:old_a] + '''def realised_share(tree, tang, run_pts, run_tan):
    """share of the GT run points that have a delivered point within TOL (ALL of them, not the nearest few: a crowded graph must not hide the right line) whose direction differs by at most ANG"""
    if tree is None:
        return 0.0
    ok = np.zeros(len(run_pts), bool)
    for i, nb in enumerate(tree.query_ball_point(run_pts, TOL)):
        if nb:
            ok[i] = bool(((tang[nb] * run_tan[i]).sum(1) >= COS).any())
    return float(ok.mean())


''' + s[old_b:]
# variants parsing: p0.5 or p>=0.5
sub('    thr = {v: float(v[3:]) for v in names if v.startswith("p>=")}\n    unknown = [v for v in names if v not in ("plain", "oracle", "oracle+fit") and v not in thr]\n    assert not unknown, f"unknown variants {unknown} (plain, p>=<t>, oracle, oracle+fit)"',
    '    names = [("p" + v[3:]) if v.startswith("p>=") else v for v in names]                      # p>=0.5 (quoted) = p0.5\n    thr = {v: float(v[1:]) for v in names if v.startswith("p") and v[1:].replace(".", "", 1).isdigit()}\n    unknown = [v for v in names if v not in ("plain", "oracle", "oracle+fit") and v not in thr]\n    assert not unknown, f"unknown variants {unknown} (plain, p<t> such as p0.5, oracle, oracle+fit)"')
# alignment + seed guards
sub('    spec = QR.parse("close:1.5@0.5")\n    per = {v: [] for v in names}',
    '    assert [d["row"] for d in dump] == [f.row for f in frames], "the full dump of B is not in the order of the VAL frames"\n    cfg = json.load(open(os.path.join(ROOT, "runs", a.run, "config.json")))\n    assert cfg.get("seed", a.seed) == a.seed, f"--seed {a.seed} does not match the run (seed {cfg.get(\'seed\')}): its keypoints would sit on another seed\'s graph"\n    spec = QR.parse("close:1.5@0.5")\n    per = {v: [] for v in names}')
sub("    for b, (f, d) in enumerate(zip(frames, dump)):\n        g = QV.load_gt(f.token)", "    for b, (f, d) in enumerate(zip(frames, dump)):\n        if a.max and b >= a.max:\n            break\n        g = QV.load_gt(f.token)")
# oracle keypoint set = valid (like the p variants); rows keep the p >= 0.5 filter
sub("        if jg is not None and any(v.startswith(\"oracle\") for v in names):\n            ma, mb = JR._match(e[\"kp_pos\"][:K], e[\"kp_dir\"][:K], ve, jg[\"ent\"]), JR._match(e[\"kp_pos\"][K:], e[\"kp_dir\"][K:], vx, jg[\"ext\"])",
    "        if jg is not None and any(v.startswith(\"oracle\") for v in names):\n            ma, mb = JR._match(e[\"kp_pos\"][:K], e[\"kp_dir\"][:K], e[\"valid\"][0], jg[\"ent\"]), JR._match(e[\"kp_pos\"][K:], e[\"kp_dir\"][K:], e[\"valid\"][1], jg[\"ext\"])")
sub("scales_or[(i, j)] = tuple(float(x) if np.isfinite(x) else 1.0 for x in sc_all[ip])", "if np.isfinite(sc_all[ip]).all():\n                        scales_or[(i, j)] = tuple(float(x) for x in sc_all[ip])")
sub('(scales_or[(int(i), int(j))] if v == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and v != "oracle") else ()))',
    '(scales_or.get((int(i), int(j)), ()) if v == "oracle+fit" else (tuple(e["scale"][i, j]) if (e.get("scale") is not None and not a.chord and v != "oracle") else ()))')
sub('    s.add_argument("--rows", action="store_true")', '    s.add_argument("--rows", action="store_true")\n    s.add_argument("--chord", action="store_true", help="the chord tangent for the p<t> variants even when the dump has learned handle scales (the control)")\n    s.add_argument("--max", type=int, default=0, help="only the first N frames (a smoke test of the pipeline)")')
sub('default=["plain", "p>=0.5", "p>=0.1", "oracle", "oracle+fit"]', 'default=["plain", "p0.5", "p0.1", "oracle", "oracle+fit"]')
open(p, "w", encoding="utf-8").write(s)
print("ok")
