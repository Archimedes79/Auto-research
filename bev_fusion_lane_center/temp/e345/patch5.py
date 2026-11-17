R = "/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/"
p = R + "jprune_table.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


# ---------------------------------------------------------------- docstring
a = s.index('"""CPU side')
b = s.index('import argparse')
s = s[:a] + '''"""CPU side of the E345 probe (D99 step 0): scoring of the point folders, the paired scene bootstrap, the halves, the classes, the greedy curves, the plan of the scored points, the controls and the table writer. The passes themselves are `jprune_probe.py`.

    wsl bash tools/env/_py14.sh jprune_table.py plan  --sweep temp/e345/e312_p_s0/sweep.json --out temp/e345/e312_p_s0/score_points.json [--runs-dir runs]      # section 3: the point sets scored in full (+ v once the handle-scale points are scored), from the sweep of SEED 0
    wsl bash tools/env/_py14.sh jprune_table.py score --run e312_p_s0 --points temp/e345/e312_p_s0/score_points.json [--only 12,101]                           # the stored scorer (`jends_eval.py score`) on the point folders of `runs/`, tag e345<NNN>
    wsl bash tools/env/_py14.sh jprune_table.py boot  --run e312_p_s0 --points temp/e345/e312_p_s0/score_points.json [--only 12,101] [--n 1000]              # the paired scene bootstrap of every scored point against the null folder on the judged read-outs
    wsl bash tools/env/_py14.sh jprune_table.py table --sweeps temp/e345/e312_p_s0,temp/e345/e312_p_s1,temp/e345/e324_s_s0,temp/e345/e324_s_s1 --out experiments/E345_pruning-sensitivity-step0 [--runs-dir runs]
    (the tests are tests/test_jprune_probe.py)

Job flow (one model, seeds 0 and 1; relative paths are relative to the problem folder): (1) GPU: `jprune_probe.py sweep` on both seeds (null twice, controls, the 84 points, random / reverse draws); (2) CPU: `plan` on the SEED-0 sweep -> `score_points.json` (18 point sets and the second null 900, without v);
(3) GPU: `sweep --score-points <that file>` again on both seeds (only the dumps are new; seed 1 may take seed 0's file; `calib.pt` keeps the channel sets); (4) CPU: `score` of the dumps, first the handle-scale points and the nulls (`--only`); (5) CPU: `plan` again (now with v, id 120); (6) GPU: `sweep --score-points` again (the dump of v);
(7) CPU: `score` of the rest; (8) CPU: `boot`; (9) `table`. `score --points` and `boot --points` read the family (E312 / E324 s) from the `sweep.json` next to the points file, so a seed-1 run may point at seed 0's `score_points.json` (the sweep.json must lie next to it: copy both, or give the seed-1 folder its own copy).
The scorer writes into `runs/` of this checkout (`jends_eval.py score` has that place built in), so `score` and `boot` have no `--runs-dir`.

Loss L is always the null's minus the ablated point's number, positive when worse (for dup_frac, where lower is better: ablated minus null). The two halves: the 22 VAL scenes in sorted order, even index = SELECTION (decides classes, fractions, the order of the curves), odd = JUDGEMENT (judges v and the
sizes); the full-VAL number is printed next to them. Classes (card section 3): a point scored in full: FREE L <= 0.005, CHEAP <= 0.02, COSTLY <= 0.05, else CRITICAL, L = topo F1 loss on the WORSE read-out and the worse seed; a point only swept: ends_mean6 with 0.003 / 0.01 / 0.03, labelled "sweep".
Bootstrap (`boot`): `jdiag.boot` (imported, not edited; 1000 resamples of the VAL scenes, seed 0, it only prints, so its lines are parsed: 4 decimals) is called with A = the ablated point and B = the null; jdiag's sign is A - B, so for the topo F1, junction F1 and the other higher-is-better numbers the
LOSS (null minus ablated) is the negative, with the interval ends swapped; for dup_frac A - B is the loss. The intervals go to `<card stem>.boot.json` next to the point's card. They are of the full-VAL loss (jdiag.boot resamples all 22 scenes; there is no half-wise interval): a point whose point-estimate class (selection half) is FREE keeps it only
when the upper end of the full-VAL loss interval is <= 0.01 on every judged read-out and seed (card section 4), else it is CHEAP; without an interval the point-estimate class stands and the label says so.
Sweep-FREE (a unit): ends_mean6, pair AP and the keypoint position error within 0.003 / 0.003 / 0.05 m, all three. Where the card is silent: the unit's FREE fraction (and v) uses the three-criterion sweep-FREE and takes the largest ladder fraction whose point is sweep-FREE; the random control compares the ranked loss with the MEAN of the three draws; v takes the
handle-scale fraction from the full class on SEED 0 (the plan runs before seed 1 is scored); the four "largest saved GMACs" slots of the plan count only the units of the two HEADS (the pyramid is a junction view that step 1 cannot deliver); an identical point set named twice (a top-4 unit that is also a tercile point) is scored once, so the plan can hold fewer than 19 sets; the second null (id 900) is scored as
well, for the spread control; the knee S_c needs the loss per GMAC to rise above twice the previous slope AND above 0.01 per GMAC (the 0.01 band), so noise does not make a knee; the "pyramid view" prefixes are cut on the MACs saved in the pyramid's own layers (4.677 GMACs), the heads-only ones on the MACs saved in the two heads (3.278);
a null point (0, 900) is scored with every read-out of the model (E324 s: snapped, PRIMARY, snap-nogeom) for the stored-card check, every other point with the two judged ones.
"""
''' + s[b:]

rep("import argparse\nimport csv\nimport json\nimport os\nimport pickle\nimport sys\n", "import argparse\nimport contextlib\nimport csv\nimport io\nimport json\nimport os\nimport pickle\nimport re\nimport sys\n")
rep("SMALL_GMACS = 0.01\n", "SMALL_GMACS = 0.01\nKNEE_FLOOR = 0.01                                                                                                                    # the 0.01 noise band per GMAC: a rise of the loss per GMAC below it is no knee\nFREE_UPPER = 0.01                                                                                                                   # a FREE class needs the upper end of the loss interval at or below this (card section 4)\nBOOT_LINE = re.compile(r\"^\\s*(\\S+)\\s+A - B: point (\\S+), bootstrap mean (\\S+), 95 % \\[(\\S+), (\\S+)\\]\")\n")

# ---------------------------------------------------------------- scoring + boot
a = s.index("# ---------------------------------------------------------------- scoring")
b = s.index("# ---------------------------------------------------------------- one model-seed")
s = s[:a] + '''# ---------------------------------------------------------------- scoring and bootstrap
def _paths(runs_dir, run, pid, opts):
    d = os.path.join(runs_dir, point_run(run, pid))
    stem = card_stem(pid, opts)
    return d, stem, os.path.join(d, stem + ".json"), os.path.join(d, stem + ".counts.pkl")


def score_point(run, family, pid, runs_dir, readouts=None):
    """the stored scorer on the point folder `runs/e345_p<NNN>_<run>/` (tag e345<NNN>, thr 0.1): a null point (0, 900) with every read-out of the model (the stored-card check needs them all), any other point with the two judged ones. A read-out that has its counts file is skipped
    (a tagged card is never overwritten); a folder without `point.json` is an unfinished dump and is refused; a card json without its counts file (an earlier scoring crashed between the two writes) is reported and left for you to move away"""
    import jends_eval as JEV
    if readouts is None:
        readouts = tuple(READOUTS[family]) if pid in (0, NULL2) else JUDGED[family]
    assert os.path.exists(os.path.join(runs_dir, point_run(run, pid), "point.json")), f"{os.path.join(runs_dir, point_run(run, pid))} has no point.json: the dump is not complete (rerun the sweep with --score-points)"
    for name in readouts:
        opts = READOUTS[family][name]
        d, stem, card, counts = _paths(runs_dir, run, pid, opts)
        if os.path.exists(counts):
            continue
        if os.path.exists(card):
            raise RuntimeError(f"{card} exists without {counts}: an earlier scoring crashed between the two writes (`qends_eval.cmd_score` writes the card first and never overwrites a tagged card). Move or delete that card json by hand, then score again")
        JEV.score(SimpleNamespace(run=point_run(run, pid), seed=seed_of(run), thr=THR, tag=f"e345{pid:03d}", **opts))


def boot_point(run, family, pid, runs_dir, n=1000, readouts=None):
    """the paired scene bootstrap of a scored point against the null folder on the judged read-outs: `jdiag.boot` with A = the ablated point, B = the null (it prints; its lines are parsed). Written as LOSS (null minus ablated; dup_frac: ablated minus null, so jdiag's A - B is
    negated except for dup_frac and the interval ends swap) to `<card stem>.boot.json` next to the point's card; an existing file is kept"""
    import jdiag
    for name in readouts or JUDGED[family]:
        opts = READOUTS[family][name]
        d, stem, _, counts = _paths(runs_dir, run, pid, opts)
        d0, stem0, _, counts0 = _paths(runs_dir, run, 0, opts)
        out = os.path.join(d, stem + ".boot.json")
        if os.path.exists(out):
            continue
        assert os.path.exists(counts) and os.path.exists(counts0), f"{counts} / {counts0}: score the point and the null first"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            jdiag.boot(d, stem, d0, stem0, n=n)                                        # an absolute run folder: `os.path.join(ROOT, "runs", <absolute>)` is the folder itself
        parsed = {m.group(1): tuple(float(m.group(i)) for i in range(2, 6)) for m in map(BOOT_LINE.match, buf.getvalue().splitlines()) if m}
        assert {"topo_F1", "junction_F1", "dup_frac"} <= set(parsed), f"jdiag.boot printed no interval lines:\\n{buf.getvalue()}"
        loss = {k: (dict(point=pt, mean=mn, lo=lo, hi=hi) if k == "dup_frac" else dict(point=-pt, mean=-mn, lo=-hi, hi=-lo)) for k, (pt, mn, lo, hi) in parsed.items()}
        PP.atomic_write(out, json.dumps(dict(point=pid, readout=name, resamples=n, scenes=22, a=f"{os.path.basename(d)}/{stem}", b=f"{os.path.basename(d0)}/{stem0}", note="loss = null minus ablated (dup_frac: ablated minus null): the sign of jdiag.boot's A - B negated, ends swapped; full-VAL loss, 4 decimals", loss=loss), indent=1).encode())


''' + s[b:]

# ---------------------------------------------------------------- RunData
rep('''    def full_loss(self, pid, half, key="topo_F1", readouts=None):
        """loss of the scored point on the WORSE read-out (None when a read-out is not scored)"""
        sign = -1.0 if key == "dup_frac" else 1.0                                       # lower is better there
        ls = []
        for r in readouts or JUDGED[self.family]:
            a, b = self.full(0, r, half, key), self.full(pid, r, half, key)
            ls.append(None if a is None or b is None else sign * (a - b))
        return None if any(x is None for x in ls) else max(ls)
''', '''    def full_loss_by_readout(self, pid, half, key="topo_F1", readouts=None):
        """-> {read-out: loss of the scored point (null minus the point; dup_frac, where lower is better: the point minus the null)}, None for a read-out that is not scored"""
        sign = -1.0 if key == "dup_frac" else 1.0
        out = {}
        for r in readouts or JUDGED[self.family]:
            a, b = self.full(0, r, half, key), self.full(pid, r, half, key)
            out[r] = None if a is None or b is None else sign * (a - b)
        return out

    def full_loss(self, pid, half, key="topo_F1", readouts=None):
        """loss of the scored point on the WORSE read-out (None when a read-out is not scored)"""
        ls = list(self.full_loss_by_readout(pid, half, key, readouts).values())
        return None if any(x is None for x in ls) else max(ls)

    def boot(self, pid, readout):
        """the bootstrap file of a point and read-out (`boot_point`), None when there is none"""
        p = os.path.join(self.runs_dir, point_run(self.run, pid), card_stem(pid, READOUTS[self.family][readout]) + ".boot.json")
        return json.load(open(p)) if os.path.exists(p) else None

    def loss_upper(self, pid, key="topo_F1"):
        """the upper end of the 95 % interval of the full-VAL loss on the WORSE judged read-out (None when one has no interval)"""
        b = [self.boot(pid, r) for r in JUDGED[self.family]]
        return None if any(x is None for x in b) else max(x["loss"][key]["hi"] for x in b)
''')

# ---------------------------------------------------------------- klass
rep('''    def klass(self, pid, half="sel"):
        """(class, label, loss) of a single point: from the full scoring when every seed has it, else from the sweep (ends_mean6); the worse seed"""
        full = [r.full_loss(pid, half) for r in self.runs]
        if all(x is not None for x in full):
            return classify(max(full), FULL_LIMITS), "full", max(full)
''', '''    def klass(self, pid, half="sel"):
        """(class, label, loss) of a single point: from the full scoring when every seed has it, else from the sweep (ends_mean6); the worse seed. A FREE point estimate holds only when the upper end of the full-VAL loss interval (`boot`) is <= 0.01 on every read-out and seed
        (label "full + interval"; without an interval the label says "point estimate only")"""
        full = [r.full_loss(pid, half) for r in self.runs]
        if all(x is not None for x in full):
            cls, loss = classify(max(full), FULL_LIMITS), max(full)
            if cls != "FREE":
                return cls, "full", loss
            ups = [r.loss_upper(pid) for r in self.runs]
            if any(u is None for u in ups):
                return cls, "full, point estimate only", loss
            return ("FREE" if max(ups) <= FREE_UPPER else "CHEAP"), "full + interval", loss
''')

# ---------------------------------------------------------------- sizes
rep('''    c = next((p[i] for i in range(1, len(slope)) if slope[i] > 2 * max(slope[i - 1], 1e-9)), None)''', '''    c = next((p[i] for i in range(1, len(slope)) if slope[i] > 2 * max(slope[i - 1], KNEE_FLOOR)), None)                  # the previous slope is floored at the 0.01 band per GMAC: a rise of noise is no knee''')
rep('''(ascending): the largest with L <= 0.01, <= 0.03, and the knee (the prefix after which the loss per saved GMAC more than doubles)"""''', '''(ascending): the largest with L <= 0.01, <= 0.03, and the knee (the prefix after which the loss per saved GMAC more than doubles the previous slope, floored at 0.01 per GMAC)"""''')

# ---------------------------------------------------------------- plan: heads only
rep('''    """the point sets scored in full (card section 3, chosen on seed 0): null, the 4 units with the largest saved GMACs among those with a sweep-FREE fraction,''', '''    """the point sets scored in full (card section 3, chosen on seed 0): null, the 4 units of the two HEADS (not the pyramid: a junction view step 1 cannot deliver) with the largest saved GMACs among those with a sweep-FREE fraction,''')
rep('''for u in model.units if model.free_fraction(u.name, [r0]) > 0]''', '''for u in model.units if u.group in ("maps", "pair") and model.free_fraction(u.name, [r0]) > 0]''')

# ---------------------------------------------------------------- spearman
rep('''def spearman(a, b):
    ra, rb = (np.argsort(np.argsort(x)) for x in (a, b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")''', '''def spearman(a, b):
    """rank correlation with the average rank for ties (nan for fewer than 3 values or a constant input)"""
    from scipy.stats import rankdata
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])''')

# ---------------------------------------------------------------- sets + table
rep('''        for r in model.runs:
            for h in ("sel", "jud", "all"):
                row[f"s{seed_of(r.run)}_L_{h}"] = r.full_loss(e["id"], h)''', '''        for r in model.runs:
            for h in ("sel", "jud", "all"):
                row[f"s{seed_of(r.run)}_L_{h}"] = r.full_loss(e["id"], h)
            row[f"s{seed_of(r.run)}_L_hi"] = r.loss_upper(e["id"])''')
rep('''            md += ["| set | units | saved GMACs | L sel s0 | L jud s0 | L sel s1 | L jud s1 |", "|---|---|---|---|---|---|---|"]
            fmt = lambda x: "-" if x is None else f"{x:+.4f}"  # noqa: E731
            md += [f"| {r['name']} | {r['units']} | {r['saved_gmacs']:.3f} | {fmt(r.get('s0_L_sel'))} | {fmt(r.get('s0_L_jud'))} | {fmt(r.get('s1_L_sel'))} | {fmt(r.get('s1_L_jud'))} |" for r in sr] + ["", f"sizes (judgement half, worse seed and read-out): {json.dumps(sz)}", ""]
        md += ["| unit | f | saved GMACs | saved Mparams | sweep L s0 (sel) | sweep L s1 (sel) | class (worse seed, sel) | label | sweep L judgement |", "|---|---|---|---|---|---|---|---|---|"]
        for p in m.runs[0].rec["points"]:
            key = f"p{p['id']:03d}"
            cl, lab, loss = m.klass(p["id"])
            d = [r.delta(key, "sel")["d6"] for r in m.runs]
            sv = m.runs[0].rec["passes"][key]
            md.append(f"| {p['unit']} | {p['f']} | {sv['saved_macs'] / 1e9:.4f} | {sv['saved_params'] / 1e6:.4f} | {d[0]:+.4f} | {d[1]:+.4f} | {cl} | {lab} | {max(r.delta(key, 'jud')['d6'] for r in m.runs):+.4f} |")
            rows.append(dict(model=name, id=p["id"], unit=p["unit"], f=p["f"], group=p["group"], saved_gmacs=sv["saved_macs"] / 1e9, saved_params=sv["saved_params"], klass=cl, label=lab, class_loss=loss,
                             **{f"s{seed_of(r.run)}_{k}_{h}": r.delta(key, h)[k] for r in m.runs for h in ("sel", "jud", "all") for k in ("d6", "dap", "dpos")}, **{f"s{seed_of(r.run)}_full_{h}": r.full_loss(p["id"], h) for r in m.runs for h in ("sel", "jud", "all")}))
        md.append("")''', '''            md += ["| set | units | saved GMACs | L sel s0 | L jud s0 | 95 % upper s0 | L sel s1 | L jud s1 | 95 % upper s1 |", "|---|---|---|---|---|---|---|---|---|"]
            fmt = lambda x: "-" if x is None else f"{x:+.4f}"  # noqa: E731
            md += [f"| {r['name']} | {r['units']} | {r['saved_gmacs']:.3f} | {fmt(r.get('s0_L_sel'))} | {fmt(r.get('s0_L_jud'))} | {fmt(r.get('s0_L_hi'))} | {fmt(r.get('s1_L_sel'))} | {fmt(r.get('s1_L_jud'))} | {fmt(r.get('s1_L_hi'))} |" for r in sr] + ["", f"sizes (judgement half, worse seed and read-out): {json.dumps(sz)}", ""]
        ro = JUDGED[m.runs[0].family]
        md += [f"| unit | f | saved GMACs | saved Mparams | sweep L s0 (sel) | sweep L s1 (sel) | full L s0 sel ({' / '.join(ro)}) | full L s1 sel ({' / '.join(ro)}) | class (worse seed, sel) | label | sweep L judgement |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        fmt = lambda x: "-" if x is None else f"{x:+.4f}"  # noqa: E731
        for p in m.runs[0].rec["points"]:
            key = f"p{p['id']:03d}"
            cl, lab, loss = m.klass(p["id"])
            d = [r.delta(key, "sel")["d6"] for r in m.runs]
            sv = m.runs[0].rec["passes"][key]
            by = [r.full_loss_by_readout(p["id"], "sel") for r in m.runs]
            md.append(f"| {p['unit']} | {p['f']} | {sv['saved_macs'] / 1e9:.4f} | {sv['saved_params'] / 1e6:.4f} | {d[0]:+.4f} | {d[1]:+.4f} | " + " | ".join(" / ".join(fmt(b[x]) for x in ro) for b in by) + f" | {cl} | {lab} | {max(r.delta(key, 'jud')['d6'] for r in m.runs):+.4f} |")
            rows.append(dict(model=name, id=p["id"], unit=p["unit"], f=p["f"], group=p["group"], saved_gmacs=sv["saved_macs"] / 1e9, saved_params=sv["saved_params"], klass=cl, label=lab, class_loss=loss,
                             **{f"s{seed_of(r.run)}_{k}_{h}": r.delta(key, h)[k] for r in m.runs for h in ("sel", "jud", "all") for k in ("d6", "dap", "dpos")}, **{f"s{seed_of(r.run)}_full_{h}": r.full_loss(p["id"], h) for r in m.runs for h in ("sel", "jud", "all")},
                             **{f"s{seed_of(r.run)}_full_{x}_{h}": r.full_loss_by_readout(p["id"], h)[x] for r in m.runs for x in ro for h in ("sel", "jud", "all")}, **{f"s{seed_of(r.run)}_full_hi": r.loss_upper(p["id"]) for r in m.runs}))
        md.append("")''')

# ---------------------------------------------------------------- main
rep('''    s = sp.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--points", required=True)
    s.add_argument("--runs-dir", default=None)
    s.add_argument("--only", default=None)
''', '''    for name in ("score", "boot"):
        s = sp.add_parser(name)
        s.add_argument("--run", required=True)
        s.add_argument("--points", required=True, help="score_points.json; the sweep.json next to it says which family (E312 / E324 s) the run is")
        s.add_argument("--only", default=None, help="comma separated point ids")
        if name == "boot":
            s.add_argument("--n", type=int, default=1000, help="resamples of the VAL scenes")
''')
rep('''    elif a.cmd == "score":
        rd = a.runs_dir or os.path.join(ROOT, "runs")
        fam = "e324" if json.load(open(os.path.join(os.path.dirname(a.points) or ".", "sweep.json")))["attach"] else "e312"
        for p in json.load(open(a.points)):
            if a.only is None or p["id"] in {int(x) for x in a.only.split(",")}:
                score_point(a.run, fam, p["id"], rd)''', '''    elif a.cmd in ("score", "boot"):
        rd = os.path.join(ROOT, "runs")
        fam = "e324" if json.load(open(os.path.join(os.path.dirname(a.points) or ".", "sweep.json")))["attach"] else "e312"
        for p in json.load(open(a.points)):
            if a.only is None or p["id"] in {int(x) for x in a.only.split(",")}:
                if a.cmd == "score":
                    score_point(a.run, fam, p["id"], rd)
                elif p["id"] not in (0, NULL2):
                    boot_point(a.run, fam, p["id"], rd, a.n)''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
