p = "tests/test_evaluate_door.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''def main():
    p = argparse.ArgumentParser()''', '''def check_sets_real(run, n, device):
    """a named set against the run's own lists, on a run trained on the split of that name (runs/t16_small_s0: split small): `--split small.val` and `--split small.train` give the
    numbers of `--split val` and `--split train` number for number (the same scenes, the same frames), under files of their own name, and the card says which set it is"""
    if not glob.glob(os.path.join(run, "ckpt_*.pt")) or not os.path.isfile(os.path.join(run, "config.json")):
        print(f"SKIP sets check: {run} has no checkpoint here")
        return False
    sys.path.insert(0, os.path.join(ROOT, "training"))
    cfg = json.load(open(os.path.join(run, "config.json")))
    import bev_sets  # noqa: E402  (evaluation/ is on the path)
    sp = bev_sets.names()
    split = next((x.split(".")[0] for x in sp if x.endswith(".val") and sorted(bev_sets.lists(x, [], [])[1]) == sorted(cfg.get("val_scenes") or [])
                  and sorted(bev_sets.lists(x, [], [])[0]) == sorted(cfg.get("train_scenes") or [])), None)
    if split is None:
        print(f"SKIP sets check: {run} was not trained on a split of splits/ (its lists are inline)")
        return False
    py, env = sys.executable, dict(os.environ, OMP_NUM_THREADS="4")
    with tempfile.TemporaryDirectory(dir=os.path.join(ROOT, "temp")) as tmp:
        jobs = {}
        for who, sset in (("val", "val"), ("named_val", f"{split}.val"), ("train", "train"), ("named_train", f"{split}.train")):
            folder = os.path.relpath(os.path.join(tmp, who, os.path.basename(run)), ROOT)
            os.makedirs(folder)
            for f in ("config.json", "gt.json"):
                if os.path.isfile(os.path.join(run, f)):
                    shutil.copy(os.path.join(run, f), folder)
            shutil.copytree(os.path.join(run, "code"), os.path.join(folder, "code"))
            jobs[who] = (folder, sset)
        for pair in (("val", "named_val"), ("train", "named_train")):                 # two at a time: each loads nuScenes and the model
            procs = []
            for who in pair:
                folder, sset = jobs[who]
                log = open(os.path.join(tmp, who + ".log"), "w")
                procs.append((who, log, subprocess.Popen([py, "evaluation/evaluate.py", folder, "--quant", "--n", str(n), "--device", device, "--workers", "0", "--ckpt-dir", run,
                                                          "--split", sset, "--out", os.path.join(folder, "suite")], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)))
            codes = [(w, pr.wait()) for w, _, pr in procs]
            for _, log, _ in procs:
                log.close()
            for w, c in codes:
                if c:
                    print(f"--- {w}.log (tail)\n" + "".join(open(os.path.join(tmp, w + ".log")).readlines()[-15:]))
            assert all(c == 0 for _, c in codes), f"exit codes {codes}"
        for plain, named, tag in (("val", "named_val", f"_{split}_val"), ("train", "named_train", f"_{split}_train")):
            pe, ne = (os.path.join(ROOT, jobs[w][0], "eval") for w in (plain, named))
            ptag = "" if plain == "val" else "_train"
            assert os.path.isfile(os.path.join(ne, f"card{tag}.json")) and not os.path.exists(os.path.join(ne, f"card{ptag}.json")), f"the set {split}.{plain[:5]} writes card{tag}.json and not the plain card"
            assert os.path.isdir(os.path.join(ne, "suite", "quant", f"eval{tag}_n{n}")), "the store carries the set's name"
            mp, mn = (json.load(open(os.path.join(e, f"metrics{t}.json"))) for e, t in ((pe, ptag), (ne, tag)))
            assert mn["split"] == f"{split}.{plain}" and mp["split"] == plain, (mn["split"], mp["split"])
            for m in (mp, mn):
                for k in ("run", "code", "split", "suite_run"):
                    m.pop(k)
            assert len(mp["metrics"]) > 500, len(mp["metrics"])
            same(mp, mn, f"metrics.json of {split}.{plain} against {plain}")
            card = json.load(open(os.path.join(ne, f"card{tag}.json")))
            use = card["provenance"]["data"]["use"]
            assert card["split"] == f"{split}.{plain}" and use["set"] == f"{split}.{plain}" and card["provenance"]["data"]["split"] == f"{split}.{plain}", (card["split"], use.get("set"))
            assert f"official split '{plain}'" in use["split_rule"], use["split_rule"]
        print(f"sets: {split}.val / {split}.train equal val / train number for number ({n} frames each, {device}), files and card named by the set")
    return True


def main():
    p = argparse.ArgumentParser()''')
rep('''    p.add_argument("--real", nargs="?", const="runs/t16_full_s0", default=None, metavar="RUN", help="also the golden check on this run")''',
    '''    p.add_argument("--real", nargs="?", const="runs/t16_full_s0", default=None, metavar="RUN", help="also the golden check on this run")
    p.add_argument("--sets", nargs="?", const="runs/t16_small_s0", default=None, metavar="RUN", help="also the named-set check on this run (trained on a split of splits/)")''')
rep('''        done += "; " + ("golden check ok" if check_golden(a.real, a.n, a.device) else "golden check skipped")''',
    '''        done += "; " + ("golden check ok" if check_golden(a.real, a.n, a.device) else "golden check skipped")
    if a.sets:
        done += "; " + ("sets check ok" if check_sets_real(a.sets, a.n, a.device) else "sets check skipped")''')
rep('''Usage (from the bev_fusion folder): python tests/test_evaluate_door.py [--real [RUN] [--n 4] [--device cpu|cuda]]''',
    '''With --sets (a run trained on a split of splits/, default runs/t16_small_s0, CPU, 4 frames, about 5 minutes): the sets of evaluation/bev_sets.py against the run's own lists:
`--split small.val` and `--split small.train` give the numbers of `--split val` / `--split train` number for number, into files named by the set (card_small_val.json, the store
eval_small_val_n4), and the card says `"split": "small.val"` with the set in its provenance.

Usage (from the bev_fusion folder): python tests/test_evaluate_door.py [--real [RUN]] [--sets [RUN]] [--n 4] [--device cpu|cuda]''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
