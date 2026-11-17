"""E291 scratch: build the STAGED copies from the base copies (the real files at HEAD 5744e2d3) by exact, single-match replacements. Re-run to rebuild the stage from scratch:  python3 apply_hunks.py
Every hunk carries a `# E291` marker in the staged file."""
import os
import shutil
import sys

R = "/mnt/c/Auto-research/bev_fusion_lane_center"
S = f"{R}/temp/_session_scratch_e291"
FILES = ("training/grid/jends.py", "training/data/lane_gt_data.py", "training/train/jends_train.py", "evaluation/jends_eval.py")


def sub(text, old, new, name):
    assert text.count(old) == 1, f"{name}: the old text occurs {text.count(old)} times: {old[:70]!r}"
    return text.replace(old, new)


def build(rel, hunks):
    t = open(f"{S}/base/{rel}", encoding="utf-8").read()
    for name, old, new in hunks:
        t = sub(t, old, new, f"{rel}:{name}")
    os.makedirs(os.path.dirname(f"{S}/stage/{rel}"), exist_ok=True)
    open(f"{S}/stage/{rel}", "w", encoding="utf-8", newline="\n").write(t)


# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ jends.py
build("training/grid/jends.py", [
    ("import", "from qlinks import SetBlock\n", "from qlinks import SetBlock\nimport jprior as JP\n"),
    ("init sig", "attach=False, paint_ev=True):\n        super().__init__()", "attach=False, paint_ev=True, prior=False):\n        super().__init__()"),
    ("init body", "        self.attach = Attach(d=d) if attach else None                                                        # E320: the attached connector ends (B's nodes as candidates), needs the node head's outputs in `forward(nodes=)`\n",
     "        self.attach = Attach(d=d) if attach else None                                                        # E320: the attached connector ends (B's nodes as candidates), needs the node head's outputs in `forward(nodes=)`\n"
     "        self.prior = JP.PriorHead(d=d) if prior else None                                                    # E291: the lane count per direction of a road-level map as a gated term on the keypoint tokens (zero at the start), needs `forward(arms=)`; built last, so the other weights draw the same numbers\n"),
    ("forward sig", "    def forward(self, jout, P, gt=None, tok=None, nodes=None):\n", "    def forward(self, jout, P, gt=None, tok=None, nodes=None, arms=None):\n"),
    ("token", "        t = self.kp(torch.cat([feat, pos / HALF, D, pr[..., None], kind[..., None]], -1))\n",
     "        t = self.kp(torch.cat([feat, pos / HALF, D, pr[..., None], kind[..., None]], -1))\n"
     "        pinfo = None\n"
     "        if self.prior is not None:                                                                                 # E291: token += sigmoid(g) * f from the arm table (detached keypoint positions, REAL peaks only: valid & real); exactly zero at the start\n"
     "            assert arms is not None, \"JEnds(prior=True) needs the arm table in `forward(arms=)` (QData.DEFAULT_PRIOR)\"\n"
     "            term, pinfo = self.prior(pos, D, valid, real, arms)\n"
     "            pinfo[\"tok\"] = t.detach().float().norm(dim=-1)\n"
     "            t = t + term.to(t.dtype)\n"),
    ("out", "        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)\n",
     "        out = dict(heat=heat, idx=idx, valid=valid, kp_pos=pos, kp_dir=D, kp_p=pr, logit=logit, allowed=allowed)\n"
     "        if pinfo is not None:                                                                                      # E291: gate, term norm, token norm, the count shown and the arm of every keypoint (the dump and the log)\n"
     "            out[\"prior\"] = pinfo\n"),
])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ lane_gt_data.py
build("training/data/lane_gt_data.py", [
    ("import", "import qzones as QZ  # noqa: E402\n", "import qzones as QZ  # noqa: E402\nimport jprior as JPR  # noqa: E402\n"),
    ("class attr", "    DEFAULT_ENDS = False                                         # E284:",
     "    DEFAULT_PRIOR = ()                                             # E291: set by `jends_train.py --lane-prior` / `jends_eval.py dump`: a tuple of `jprior.Spec`, the batch carries one arm table per spec (`b['arms']`, a list; needs the junction GT)\n"
     "    DEFAULT_ENDS = False                                         # E284:"),
    ("init", "        self.ppts = QData.DEFAULT_PPTS if ppts is None else bool(ppts)\n",
     "        self.ppts = QData.DEFAULT_PPTS if ppts is None else bool(ppts)\n"
     "        self.prior, self._ptok, self._pjg = tuple(QData.DEFAULT_PRIOR), None, (None, None)               # E291\n"),
    ("assert", "        assert not self.zone or self.sparse is not None, \"the zone maps need the sparse node target\"\n",
     "        assert not self.zone or self.sparse is not None, \"the zone maps need the sparse node target\"\n"
     "        assert not self.prior or self.junction, \"the arm table is taken from the junction GT: DEFAULT_JUNCTION\"\n"),
    ("method", "    def _open(self):\n",
     "    def partner_arms(self, token):                                                                         # E291 read-out S: the junction GT of ANOTHER frame, the first with arms from position (index + N // 2) of this split on (`jprior.partners`)\n"
     "        if self._ptok is None:\n"
     "            self._ptok = [self.tokens[int(self.bd.rows[k])] for k in self.ks]\n"
     "        if self._pjg[0] == token:                                                                          # the read-outs S and SC of one frame share the partner\n"
     "            return self._pjg[1]\n"
     "        for tk in JPR.partners(token, self._ptok):\n"
     "            jg = QJ.JG.junction_gt(self.lg.load(tk))\n"
     "            if jg is not None:\n"
     "                self._pjg = (token, jg)\n"
     "                return jg\n"
     "        self._pjg = (token, None)\n"
     "        return None\n"
     "\n"
     "    def _open(self):\n"),
    ("getitem", "                out[\"ppts\"] = torch.from_numpy(QJ.pair_polylines(jg)) if (jg is not None and len(jg[\"pairs\"]) == npair) else torch.full((npair, QJ.PPTS_N, 2), float(\"nan\"))\n",
     "                out[\"ppts\"] = torch.from_numpy(QJ.pair_polylines(jg)) if (jg is not None and len(jg[\"pairs\"]) == npair) else torch.full((npair, QJ.PPTS_N, 2), float(\"nan\"))\n"
     "            if self.prior:                                                                                 # E291: the arm table of the junction GT (c, t, n_in, n_out only), one per spec: a pure function of (token, zone, rate)\n"
     "                out[\"arms\"] = [JPR.table(jg, token, sp, self.partner_arms) for sp in self.prior]\n"),
    ("collate", "    L = max(1, max(len(o[\"lk\"]) for o in b))\n",
     "    if \"arms\" in b[0]:                                                                                  # E291: the arm tables, padded: one collated dict per spec\n"
     "        out[\"arms\"] = [JPR.collate([o[\"arms\"][k] for o in b]) for k in range(len(b[0][\"arms\"]))]\n"
     "    L = max(1, max(len(o[\"lk\"]) for o in b))\n"),
])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ jends_train.py
build("training/train/jends_train.py", [
    ("import", "import jends as JE  # noqa: E402\n", "import jends as JE  # noqa: E402\nimport jprior as JP  # noqa: E402\n"),
    ("model sig", "scales=False, e2e=False, ntok=False, unc=False, attach=False):\n        super().__init__()", "scales=False, e2e=False, ntok=False, unc=False, attach=False, prior=False):\n        super().__init__()"),
    ("model je", "unc=unc, attach=attach, paint_ev=\"nopaint\" not in maps_args.get(\"ctx\", \"dil\"))", "unc=unc, attach=attach, paint_ev=\"nopaint\" not in maps_args.get(\"ctx\", \"dil\"), prior=prior)"),
    ("forward sig", "    def forward(self, x, gt=None):\n", "    def forward(self, x, gt=None, arms=None):\n"),
    ("forward call", "        return P, out, self.je(out, P, gt=gt, tok=tok, nodes=nd if self.je.attach is not None else None)\n",
     "        return P, out, self.je(out, P, gt=gt, tok=tok, nodes=nd if self.je.attach is not None else None, arms=arms)\n"),
    ("flags", "    ap.add_argument(\"--e2e\", action=\"store_true\", help=\"E305:",
     "    ap.add_argument(\"--lane-prior\", action=\"store_true\", help=\"E291: the number of lanes per direction of a road-level map (the arm table of the junction GT: centre, heading, count of entries and exits) enters the keypoint tokens as a gated term (zero at the start)\")\n"
     "    ap.add_argument(\"--prior-noise\", default=\"0\", choices=(\"0\", \"mix\"), help=\"E291: 0 = clean counts (20 %% of the junctions have no map); mix = the rate of wrong counts of each junction from {0, 0.1, 0.3} by a hash of (frame, zone), 15 %% of the arms unknown, 20 %% of the junctions without a map, 20 %% with all counts unknown\")\n"
     "    ap.add_argument(\"--e2e\", action=\"store_true\", help=\"E305:"),
    ("asserts", "    T.seed_all(a.seed)\n    if a.raster == \"exact\":",
     "    assert a.lane_prior or a.prior_noise == \"0\", \"--prior-noise mix needs --lane-prior\"\n"
     "    T.seed_all(a.seed)\n    if a.raster == \"exact\":"),
    ("data", "    LD.QData.DEFAULT_PPTS = a.geom_w > 0                                                                 # E317: the GT connector polylines of the true pairs, only for the geometry loss\n",
     "    LD.QData.DEFAULT_PPTS = a.geom_w > 0                                                                 # E317: the GT connector polylines of the true pairs, only for the geometry loss\n"
     "    LD.QData.DEFAULT_PRIOR = (JP.Spec.train(a.prior_noise),) if a.lane_prior else ()                    # E291: one arm table per batch (none without the flag), corrupted in the data path by a pure function of (token, zone)\n"),
    ("sigma", "    ck = torch.load(mp, map_location=\"cpu\")\n    model = JEndModel(",
     "    ck = torch.load(mp, map_location=\"cpu\")\n"
     "    sig = float(os.environ.get(\"BFLC_SIG\", \"1.5\"))                                                      # E291: the heat targets of the joint training must have the width of the maps checkpoint (a launcher without BFLC_SIG would train 1.5 m targets on 1.0 m maps)\n"
     "    assert abs(sig - float(ck[\"args\"].get(\"sigma\", 1.5))) < 1e-9, f\"BFLC_SIG={sig} but the maps checkpoint {mp} was trained with sigma {ck['args'].get('sigma', 1.5)}: export BFLC_SIG={ck['args'].get('sigma', 1.5)}\"\n"
     "    model = JEndModel("),
    ("model build", "e2e=a.e2e, ntok=a.ntok, unc=a.unc, attach=a.attach)\n    model.jmodel.load_state_dict(ck[\"state\"])", "e2e=a.e2e, ntok=a.ntok, unc=a.unc, attach=a.attach, prior=a.lane_prior)\n    model.jmodel.load_state_dict(ck[\"state\"])"),
    ("log line", "unc={a.unc} attach={a.attach}: {steps} steps", "unc={a.unc} attach={a.attach} prior={a.prior_noise if a.lane_prior else False}: {steps} steps"),
    ("loss in", "        _, jout, out = m(x, gt=gt)\n",
     "        arms = {k: v.to(dev) for k, v in b[\"arms\"][0].items()} if a.lane_prior else None                    # E291\n"
     "        _, jout, out = m(x, gt=gt, arms=arms)\n"),
    ("loss out", "        calls[0] += 1\n",
     "        calls[0] += 1\n"
     "        if a.lane_prior:                                                                                   # E291: the size of the prior term against the token and its gate (the kill criterion: the term must not be zero on VAL); the gradient of the previous step is still in .grad here (the trainer's zero_grad comes after this call)\n"
     "            pi, vm = out[\"prior\"], out[\"valid\"].flatten(1).float()\n"
     "            info[\"prior_ratio\"] = float((pi[\"term\"] * vm).sum() / (pi[\"tok\"] * vm).sum().clamp_min(1e-9))\n"
     "            info[\"prior_gate\"] = float((pi[\"gate\"] * vm).sum() / vm.sum().clamp_min(1.0))\n"
     "            if calls[0] in (2, 3, 4, 21):\n"
     "                gn = lambda mod: sum(float(p.grad.float().pow(2).sum()) for p in mod.parameters() if p.grad is not None) ** 0.5  # noqa: E731\n"
     "                run.log(f\"prior gradient norm after step {calls[0] - 1} (clipped with the rest): feat {gn(m.je.prior.feat):.3e}, gate {gn(m.je.prior.gate):.3e} (the gate's is zero at step 1: its gradient is proportional to f, which starts at zero); term / token {info['prior_ratio']:.3e}, gate {info['prior_gate']:.3f}\")\n"),
    ("config", "unc=a.unc, attach=a.attach, maps=mp))", "unc=a.unc, attach=a.attach, maps=mp, **({\"lane_prior\": a.prior_noise} if a.lane_prior else {})))"),
])

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ jends_eval.py
OLD_DUMP = open(f"{S}/base/evaluation/jends_eval.py", encoding="utf-8").read()
i0, i1 = OLD_DUMP.index("def dump(a):"), OLD_DUMP.index("def score(a):")
NEW_DUMP = '''def dump(a):
    import lane_gt_data as LD
    import qendnet as QEN
    import jprior as JP
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":                                                                                                                          # D100: a dump / score next to two trainings crashed one of them (CUDA unknown error, 2026-11-14)
        import gpu_gate
        gpu_gate.wait_for_room(who=f"jends_eval dump {a.run}")
    m, args = load_model(a.run, dev)
    pmode, pnoise = getattr(a, "prior_mode", None), getattr(a, "prior_noise", 0.0)                                                              # (a caller with its own Namespace has neither)
    specs = JP.readouts(pmode or "clean", pnoise) if m.je.prior is not None else []                                              # E291: a model with the lane prior has one read-out per spec, all from ONE pass over the loader
    assert m.je.prior is not None or (pmode is None and not pnoise), f"{a.run} has no lane prior: --prior-mode / --prior-noise do not apply"
    run_dirs = [os.path.join(ROOT, "runs", f"{a.run}_{sp.name}" + (f"_max{a.max}" if a.max else "")) for sp in specs] or [os.path.join(ROOT, "runs", a.run)]         # E291: a read-out is a NEW run folder (a --max test has its own names), never an overwritten one
    for d in run_dirs if specs else []:                                                                                                        # all names are checked first, the folders are made when the pass is done (a crash leaves none)
        assert not os.path.exists(d), f"{d} exists: a read-out folder is never overwritten"
    LD.QData.DEFAULT_JUNCTION = True
    LD.QData.DEFAULT_PRIOR = tuple(specs)
    ds, dl = LD.loader("val_clean", bs=8, workers=2, shuffle=False, map_mode="error", pixels=False)
    assert ds.missing == 0, f"{ds.missing} VAL frames have no stored lane GT"
    R = max(1, len(specs))
    frames, tot, psum = [[] for _ in range(R)], [{} for _ in range(R)], [dict(term=0.0, tok=0.0, gate=0.0, n=0.0, arm=0.0) for _ in range(R)]
    with torch.no_grad():
        for b in dl:
            x = b["x"].to(dev).float()
            outs = [m(x, gt=None, arms=({k: v.to(dev) for k, v in b["arms"][r].items()} if specs else None))[2] for r in range(R)]
            for r, out in enumerate(outs):
                for k, v in QEN.end_counts(out, b, a.thr).items():
                    tot[r][k] = tot[r].get(k, 0) + v
            if a.max and len(frames[0]) >= a.max:
                break
            for r, out in enumerate(outs):
                if specs:                                                                                                                      # E291: the size of the prior term and the gate over the valid keypoints
                    vm = out["valid"].flatten(1).float()
                    psum[r]["term"] += float((out["prior"]["term"] * vm).sum())
                    psum[r]["tok"] += float((out["prior"]["tok"] * vm).sum())
                    psum[r]["gate"] += float((out["prior"]["gate"] * vm).sum())
                    psum[r]["n"] += float(vm.sum())
                    psum[r]["arm"] += float(((out["prior"]["arm"] >= 0).float() * vm).sum())                                                  # the valid keypoints that have an arm of the table in reach (S: the partner's arms lie in the partner's frame)
                for i in range(len(b["token"])):
                    fr = dict(token=b["token"][i], row=int(b["row"][i]), kp_pos=out["kp_pos"][i].cpu().numpy(), kp_dir=out["kp_dir"][i].cpu().numpy(), kp_p=out["kp_p"][i].cpu().numpy(),
                              valid=out["valid"][i].cpu().numpy(), logit=out["logit"][i].float().cpu().numpy(), allowed=out["allowed"][i].cpu().numpy(),
                              scale=out["scale"][i].float().cpu().numpy() if "scale" in out else None)
                    if "kp_logsig" in out:                                                                                                         # E318: the position scale (log sigma, m) of every keypoint
                        fr["logsig"] = out["kp_logsig"][i].float().cpu().numpy()
                    if "att_pos" in out:                                                                                                           # E320: the attached ends (position, heading, probability of an attachment) of every keypoint
                        fr.update(att_pos=out["att_pos"][i].float().cpu().numpy(), att_dir=out["att_dir"][i].float().cpu().numpy(), att_p=out["att_p"][i].float().cpu().numpy())
                    if specs:                                                                                                                      # E291: per keypoint the gate, the norm of the term and of the token, the count that was shown (-1 = unknown or absent) and its arm
                        fr["prior"] = {k: out["prior"][k][i].cpu().numpy() for k in ("gate", "term", "tok", "n", "arm")}
                    frames[r].append(fr)
    rr = lambda x, y: x / max(y, 1)  # noqa: E731
    for r, run_dir in enumerate(run_dirs):
        head = dict(ent_recall=rr(tot[r]["ent_hit"], tot[r]["ent_n"]), ent_precision=rr(tot[r]["ent_ok"], tot[r]["ent_kp"]), ext_recall=rr(tot[r]["ext_hit"], tot[r]["ext_n"]), ext_precision=rr(tot[r]["ext_ok"], tot[r]["ext_kp"]),
                    pair_recall=rr(tot[r]["pairs_hit"], tot[r]["pairs_n"]), pair_precision=rr(tot[r]["pairs_ok"], tot[r]["pairs_kp"]), counts=tot[r], thr=a.thr, frames=len(frames[r]), order=not args["no_order"],
                    evidence=not args["no_evidence"], maps=args["maps"])
        if specs:                                                                                                                              # E291: the read-out and the size of the prior term: sum of ||sigmoid(g) f|| over the sum of ||t|| of the valid keypoints (the kill criterion: not zero), the mean gate
            head["prior"] = dict(readout=specs[r].name, spec=dataclasses.asdict(specs[r]), term_over_token=psum[r]["term"] / max(psum[r]["tok"], 1e-9), gate_mean=psum[r]["gate"] / max(psum[r]["n"], 1.0), arm_share=psum[r]["arm"] / max(psum[r]["n"], 1.0), source_run=a.run)
            print(f"{specs[r].name}: {head['prior']['arm_share']:.3f} of the valid keypoints have an arm of the table in reach")
            os.makedirs(run_dir, exist_ok=False)                                                                                               # the folder is made now: a copy of the model and its config, then the dump and the cards of its own
            for f in ("model.pt", "config.json"):
                shutil.copy2(os.path.join(ROOT, "runs", a.run, f), os.path.join(run_dir, f))
        pickle.dump(frames[r], open(os.path.join(run_dir, "ends_val.pkl"), "wb"))
        json.dump(head, open(os.path.join(run_dir, "ends_head.json"), "w"), indent=1)
        print(run_dir, json.dumps({k: v for k, v in head.items() if k != "counts"}, indent=1))


'''
EV = OLD_DUMP[:i0] + NEW_DUMP + OLD_DUMP[i1:]
for name, old, new in [
    ("import", "import argparse\nimport json\n", "import argparse\nimport dataclasses\nimport json\n"),
    ("import2", "import pickle\nimport sys\n", "import pickle\nimport shutil\nimport sys\n"),
    ("load", "attach=bool(a.get(\"attach\", False)))\n    m.load_state_dict(ck[\"state\"])", "attach=bool(a.get(\"attach\", False)), prior=bool(a.get(\"lane_prior\", False)))\n    m.load_state_dict(ck[\"state\"])"),
    ("flags", "            s.add_argument(\"--max\", type=int, default=0, help=\"only the first N frames (a test of the pipeline; the score needs all 873)\")\n",
     "            s.add_argument(\"--max\", type=int, default=0, help=\"only the first N frames (a test of the pipeline; the score needs all 873)\")\n"
     "            s.add_argument(\"--prior-mode\", default=None, help=\"E291, a model trained with --lane-prior: the read-outs of the arm table, a comma list of clean (default), noise10, noise30, pm1, u, s or `noise` with --prior-noise r: ONE pass over the loader, one NEW run folder <run>_<name> per read-out (model.pt and config.json copied, ends_val.pkl, ends_head.json); u = every count unknown, geometry kept; s = the table of another frame\")\n"
     "            s.add_argument(\"--prior-noise\", type=float, default=0.0, help=\"E291: the rate of wrong counts of --prior-mode noise (0.1 -> folder <run>_noise10)\")\n"),
]:
    EV = sub(EV, old, new, f"jends_eval:{name}")
# the selftest: the pair head with the prior as the evaluation builds it
EV = sub(EV, "    print(\"jends_eval selftest ok\")\n",
         "    mp = JT.JEndModel(dict(d=64, ctx=\"none\"), order=True, evidence=True, prior=True)                                                           # E291: the prior head as the dump builds it: zero term at the start, the table in, the numbers of the prior out\n"
         "    import jprior as JP\n"
         "    import numpy as np\n"
         "    arms = JP.collate([dict(c=np.zeros((1, 2), \"float32\"), t=np.array([[0.0, 1.0]], \"float32\"), n_in=np.array([2]), n_out=np.array([2]))])\n"
         "    _, _, op = mp(torch.randn(1, 39, 200, 200), arms=arms)\n"
         "    assert op[\"prior\"][\"term\"].shape == (1, 64) and float(op[\"prior\"][\"term\"].abs().max()) == 0.0 and op[\"prior\"][\"tok\"].shape == (1, 64), \"the prior term is zero at the start\"\n"
         "    print(\"jends_eval selftest ok\")\n", "jends_eval:selftest")
os.makedirs(f"{S}/stage/evaluation", exist_ok=True)
open(f"{S}/stage/evaluation/jends_eval.py", "w", encoding="utf-8", newline="\n").write(EV)
print("stage built:", ", ".join(FILES))
