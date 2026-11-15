"""E292 scratch: scorer plumbing (--full / --base / --spec / --tag, refuse to overwrite a tagged card, pyramid assert) as patched COPIES in temp/_session_scratch_s292/new/.
usage (from the problem folder): python3 temp/_session_scratch_s292/patch_scorer.py"""
import os

NEW = "temp/_session_scratch_s292/new"


def sub(path, pairs):
    s = open(path, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (path, a[:80], s.count(a))
        s = s.replace(a, b)
    open(os.path.join(NEW, os.path.basename(path)), "w", encoding="utf-8", newline="\n").write(s)


# ---- evaluation/qends_eval.py
sub("evaluation/qends_eval.py", [
    ("import pickle\nimport sys\nimport time\n", "import pickle\nimport re\nimport sys\nimport time\n"),
    # the helper before cmd_score
    ("def cmd_score(a):\n    import torch\n",
     'DEFAULT_SPEC = "close:1.5@0.5"                                                                                                  # the read-out of B\'s delivered graph since E275\n'
     "SNAP_OPTS = (\"snap_all\", \"snap_guard\", \"snap_dedup\", \"snap_nogeom\", \"snap_both\")\n\n\n"
     "def card_name(a):\n"
     "    \"\"\"the file name of the card of this scoring (`a`: the options of `cmd_score`). The card of record of a run (`ends_card.json`: the unsnapped read-out at 0.5) is only written by the plain call; every other\n"
     "    option gets its own file (the E308 incident of 2026-11-12); a scoring of ANOTHER head's graph (`--full` / `--base` / `--spec`, E292) carries its `--tag` in the name.\"\"\"\n"
     "    snap, chord, attach, attach_r = bool(getattr(a, \"snap\", False)), bool(getattr(a, \"chord\", False)), bool(getattr(a, \"attach\", False)), getattr(a, \"attach_r\", None)\n"
     "    opts = [k for k in SNAP_OPTS if getattr(a, k, False)]\n"
     "    tag = getattr(a, \"tag\", None)\n"
     "    if not snap and not chord and not attach and a.thr == 0.5:\n"
     "        return \"ends_card.json\" if not tag else f\"ends_card_{tag}.json\"\n"
     "    return (\"ends_card_\" + (tag + \"_\" if tag else \"\") + (\"chord_\" if chord else \"\") + (\"snap\" + \"\".join(\"_\" + k[5:] for k in opts) + \"_\" if snap else \"\")\n"
     "            + ((\"attach_\" + (f\"r{attach_r}_\" if attach_r else \"\")) if attach else \"\") + f\"thr{a.thr}.json\")\n\n\n"
     "def score_target(a, run_dir):\n"
     "    \"\"\"-> (card name, read-out spec, tag) after the checks of E292: a scoring of another head's graph (`--full` the full prediction dump, `--base` the checkpoint of the same head, `--spec` the read-out) needs a `--tag`\n"
     "    that names the card (the card of record of the run is never reached), `--full` and `--base` go together, and a TAGGED card or counts file that exists is never overwritten (give a new tag; the untagged\n"
     "    re-scoring of the run's own graph stays as it was).\"\"\"\n"
     "    tag = getattr(a, \"tag\", None)\n"
     "    full, base, spec = getattr(a, \"full\", None), getattr(a, \"base\", None), getattr(a, \"spec\", None)\n"
     "    assert not (full or base or spec) or tag, \"--full / --base / --spec score another graph than B's: give a --tag (it names the card, so that the card of record of the run stays)\"\n"
     "    assert bool(full) == bool(base), \"--full and --base go together: the dump and the checkpoint of the same head\"\n"
     "    assert tag is None or re.fullmatch(r\"[a-z0-9]+\", tag), f\"--tag {tag!r}: lower-case letters and digits only (it is part of a file name)\"\n"
     "    name = card_name(a)\n"
     "    if tag:\n"
     "        for f in (name, name[:-5] + \".counts.pkl\"):\n"
     "            assert not os.path.exists(os.path.join(run_dir, f)), f\"{os.path.join(run_dir, f)} exists: a tagged card is never overwritten (give a new --tag or move the old file by hand)\"\n"
     "    return name, spec or DEFAULT_SPEC, tag\n\n\n"
     "def cmd_score(a):\n    import torch\n"),
    # the run dir exists first; the target is checked before the (long) scoring
    ("    run_dir = os.path.join(ROOT, \"runs\", a.run)\n    base_pt = a.base or os.path.join(ROOT, \"runs\", f\"e273_p_s{a.seed}\", \"model.pt\")\n",
     "    run_dir = os.path.join(ROOT, \"runs\", a.run)\n    name, spec_s, tag = score_target(a, run_dir)                                                                        # before anything long runs\n    base_pt = a.base or os.path.join(ROOT, \"runs\", f\"e273_p_s{a.seed}\", \"model.pt\")\n"),
    ("    spec = QR.parse(\"close:1.5@0.5\")\n    rec = dict(base=[], head=[])\n", "    spec = QR.parse(spec_s)\n    rec = dict(base=[], head=[])\n"),
    # the old name computation is gone (card_name), the card carries what it was made from
    ("    opts = [k for k in (\"snap_all\", \"snap_guard\", \"snap_dedup\", \"snap_nogeom\", \"snap_both\") if getattr(a, k, False)]\n    out[\"snap_options\"] = opts\n",
     "    opts = [k for k in SNAP_OPTS if getattr(a, k, False)]\n    out[\"snap_options\"] = opts\n"
     "    out[\"full\"] = os.path.relpath(full, ROOT).replace(os.sep, \"/\") if full.startswith(ROOT) else full                    # E292: the dump the delivered graph was read from, the read-out and the tag are part of the card\n"
     "    out[\"spec\"] = spec_s\n"
     "    out[\"tag\"] = tag\n"),
    ("    chord = bool(getattr(a, \"chord\", False))\n    name = \"ends_card.json\" if (not snap and not chord and not attach and a.thr == 0.5) else \"ends_card_\" + (\"chord_\" if chord else \"\") + (\"snap\" + \"\".join(\"_\" + k[5:] for k in opts) + \"_\" if snap else \"\") + ((\"attach_\" + (f\"r{attach_r}_\" if attach_r else \"\")) if attach else \"\") + f\"thr{a.thr}.json\"     # the chord control has its own file too (it overwrote the card of the learned scales on 2026-11-12)\n",
     "    # the file name is `card_name(a)` (the chord control has its own file too: it overwrote the card of the learned scales on 2026-11-12)\n"),
    ("    s.add_argument(\"--full\", default=None)\n    s.add_argument(\"--base\", default=None)\n",
     "    s.add_argument(\"--full\", default=None, help=\"the full prediction dump of the head whose graph is delivered (default: B's, temp/e276/predfull_e273_p_s<seed>.pkl); E292: the dump of B'; needs --base and --tag\")\n"
     "    s.add_argument(\"--base\", default=None, help=\"the checkpoint of the same head (default runs/e273_p_s<seed>/model.pt); needs --full\")\n"
     "    s.add_argument(\"--spec\", default=None, help=\"the read-out of the delivered graph (qgraph_readout spec; default close:1.5@0.5, the hard close-node step; E292: plain@0.5 for a head with the layer); needs --tag\")\n"
     "    s.add_argument(\"--tag\", default=None, help=\"E292: names the card (ends_card_<tag>_...): mandatory with --full / --base / --spec; a tagged card is never overwritten\")\n"),
    # selftest
    ("    print(\"qends_eval selftest ok\")\n",
     "    # E292: the card names, the tag rules and the refusal to overwrite a tagged card\n"
     "    from types import SimpleNamespace as NS\n"
     "    import tempfile\n"
     "    names = [(NS(thr=0.5), \"ends_card.json\"), (NS(thr=0.1, snap=True), \"ends_card_snap_thr0.1.json\"), (NS(thr=0.1, snap=True, snap_nogeom=True), \"ends_card_snap_nogeom_thr0.1.json\"),\n"
     "             (NS(thr=0.5, chord=True), \"ends_card_chord_thr0.5.json\"), (NS(thr=0.1, attach=True), \"ends_card_attach_thr0.1.json\"), (NS(thr=0.1, attach=True, attach_r=1.5), \"ends_card_attach_r1.5_thr0.1.json\"),\n"
     "             (NS(thr=0.5, tag=\"a\"), \"ends_card_a.json\"), (NS(thr=0.1, snap=True, tag=\"c2\"), \"ends_card_c2_snap_thr0.1.json\")]\n"
     "    for opt, want in names:\n"
     "        assert card_name(opt) == want, (card_name(opt), want)\n"
     "    with tempfile.TemporaryDirectory() as d:\n"
     "        def refused(opt):\n"
     "            try:\n"
     "                score_target(opt, d)\n"
     "            except AssertionError:\n"
     "                return True\n"
     "            return False\n"
     "        assert score_target(NS(thr=0.1, snap=True), d) == (\"ends_card_snap_thr0.1.json\", DEFAULT_SPEC, None)\n"
     "        assert score_target(NS(thr=0.1, snap=True, full=\"x.pkl\", base=\"b.pt\", spec=\"plain@0.5\", tag=\"a\"), d) == (\"ends_card_a_snap_thr0.1.json\", \"plain@0.5\", \"a\")\n"
     "        assert refused(NS(thr=0.1, snap=True, full=\"x.pkl\", base=\"b.pt\")), \"another head's dump without a tag must be refused\"\n"
     "        assert refused(NS(thr=0.1, snap=True, spec=\"plain@0.5\")), \"another read-out without a tag must be refused\"\n"
     "        assert refused(NS(thr=0.1, snap=True, full=\"x.pkl\", tag=\"a\")), \"--full without --base must be refused\"\n"
     "        assert refused(NS(thr=0.1, snap=True, tag=\"A b\")), \"a tag that is not a file-name word must be refused\"\n"
     "        for f in (\"ends_card_a_snap_thr0.1.json\",):\n"
     "            open(os.path.join(d, f), \"w\").write(\"{}\")\n"
     "        assert refused(NS(thr=0.1, snap=True, full=\"x.pkl\", base=\"b.pt\", tag=\"a\")), \"an existing tagged card must not be overwritten\"\n"
     "        os.remove(os.path.join(d, \"ends_card_a_snap_thr0.1.json\"))\n"
     "        open(os.path.join(d, \"ends_card_a_snap_thr0.1.counts.pkl\"), \"w\").write(\"x\")\n"
     "        assert refused(NS(thr=0.1, snap=True, full=\"x.pkl\", base=\"b.pt\", tag=\"a\")), \"an existing tagged counts file must not be overwritten\"\n"
     "        open(os.path.join(d, \"ends_card_snap_thr0.1.json\"), \"w\").write(\"{}\")\n"
     "        assert not refused(NS(thr=0.1, snap=True)), \"the untagged re-scoring of the run's own graph keeps overwriting its card (as before E292)\"\n"
     "    print(\"qends_eval selftest ok\")\n"),
])

# ---- evaluation/jends_eval.py
sub("evaluation/jends_eval.py", [
    ("    base = torch.load(os.path.join(ROOT, \"runs\", f\"e273_p_s{a.seed}\", \"model.pt\"), map_location=\"cpu\")[\"state\"]\n",
     "    base_pt = getattr(a, \"base\", None) or os.path.join(ROOT, \"runs\", f\"e273_p_s{a.seed}\", \"model.pt\")                          # E292: the checkpoint of the head whose graph is delivered (B')\n"
     "    base = torch.load(base_pt, map_location=\"cpu\")[\"state\"]\n"),
    ("    print(\"pyramid of the maps run == the base run's:\", pyr_same, \"(False = a joint pyramid: the delivered graph is B's, the connectors come from other pixel features: label the result 'maps-only')\")\n",
     "    print(\"pyramid of the maps run == the base run's:\", pyr_same, \"(False = a joint pyramid: the delivered graph is B's, the connectors come from other pixel features: label the result 'maps-only')\")\n"
     "    assert pyr_same or not getattr(a, \"base\", None), \"E292: the pyramid of the maps run is not the pyramid of --base: the connectors would sit on other pixel features than the delivered graph\"\n"),
    ("QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=None, base=None, skip_base_check=True,",
     "QE.cmd_score(SimpleNamespace(run=a.run, seed=a.seed, thr=a.thr, full=getattr(a, \"full\", None), base=getattr(a, \"base\", None), spec=getattr(a, \"spec\", None), tag=getattr(a, \"tag\", None), skip_base_check=True,"),
    ("            s.add_argument(\"--attach\", action=\"store_true\"",
     "            s.add_argument(\"--full\", default=None, help=\"E292: the full prediction dump of the head whose graph is delivered (B'); needs --base and --tag\")\n"
     "            s.add_argument(\"--base\", default=None, help=\"E292: the checkpoint of that head; its pyramid must be the maps run's (asserted); needs --full\")\n"
     "            s.add_argument(\"--spec\", default=None, help=\"E292: the read-out of the delivered graph (default close:1.5@0.5; plain@0.5 for a head with the Matrix-NMS layer); needs --tag\")\n"
     "            s.add_argument(\"--tag\", default=None, help=\"E292: names the card (ends_card_<tag>_...): mandatory with --full / --base / --spec; a tagged card is never overwritten\")\n"
     "            s.add_argument(\"--attach\", action=\"store_true\""),
])
print("patched copies:", sorted(os.listdir(NEW)))
