p = "tests/test_evaluate_door.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    assert s.count(old) == n, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('const="runs/t16_small_s0"', 'const="runs/t16_full_s0"')
rep("(a run trained on a split of splits/, default runs/t16_small_s0, CPU, 4 frames, about 5 minutes)", "(a run trained on a split of splits/, default runs/t16_full_s0 = large, CPU, 4 frames, about 6 minutes)")
rep('''`--split small.val` and `--split small.train` give the numbers of `--split val` / `--split train` number for number, into files named by the set (card_small_val.json, the store
eval_small_val_n4), and the card says `"split": "small.val"` with the set in its provenance.''',
    '''`--split large.val` and `--split large.train` give the numbers of `--split val` / `--split train` number for number, into files named by the set (card_large_val.json, the store
eval_large_val_n4), and the card says `"split": "large.val"` with the set in its provenance; and a set that is not the run's own (`small.val`, 10 of its 82 val scenes) evaluates
exactly those scenes, under its own name.''')
rep('''    """a named set against the run's own lists, on a run trained on the split of that name (runs/t16_small_s0: split small): `--split small.val` and `--split small.train` give the
    numbers of `--split val` and `--split train` number for number (the same scenes, the same frames), under files of their own name, and the card says which set it is"""''',
    '''    """a named set against the run's own lists, on a run trained on the split of that name (runs/t16_full_s0: split large): `--split large.val` and `--split large.train` give the
    numbers of `--split val` and `--split train` number for number (the same scenes, the same frames), under files of their own name, and the card says which set it is; and a
    set that is not the run's own (small.val inside large.val) evaluates exactly its scenes"""''')
rep('''        print(f"sets: {split}.val / {split}.train equal val / train number for number ({n} frames each, {device}), files and card named by the set")
    return True''',
    '''        other = "small.val"                                                           # a set the run was not trained on: its scenes, and only its scenes
        extra = ""
        if other in sp and split != "small":
            folder = os.path.relpath(os.path.join(tmp, "other", os.path.basename(run)), ROOT)
            os.makedirs(folder)
            for f in ("config.json", "gt.json"):
                if os.path.isfile(os.path.join(run, f)):
                    shutil.copy(os.path.join(run, f), folder)
            shutil.copytree(os.path.join(run, "code"), os.path.join(folder, "code"))
            code = subprocess.call([py, "evaluation/evaluate.py", folder, "--quant", "--n", "10", "--device", device, "--workers", "0", "--ckpt-dir", run, "--split", other,
                                    "--out", os.path.join(folder, "suite")], cwd=ROOT, env=env, stdout=open(os.path.join(tmp, "other.log"), "w"), stderr=subprocess.STDOUT)
            if code:
                print("--- other.log (tail)\n" + "".join(open(os.path.join(tmp, "other.log")).readlines()[-15:]))
            assert code == 0, f"exit code {code}"
            card = json.load(open(os.path.join(ROOT, folder, "eval", "card_small_val.json")))
            used = set(card["provenance"]["data"]["use"]["scene_names"])
            want = set(bev_sets.lists(other, [], [])[1])
            assert used and used <= want and card["split"] == other, (sorted(used), card["split"])
            assert card["provenance"]["data"]["frame_hash"] != json.load(open(os.path.join(ROOT, jobs["named_val"][0], "eval", f"card_{split}_val.json")))["provenance"]["data"]["frame_hash"], "another set, other frames"
            extra = f"; {other} evaluated {len(used)} of its {len(want)} scenes, all of them its own"
        print(f"sets: {split}.val / {split}.train equal val / train number for number ({n} frames each, {device}), files and card named by the set{extra}")
    return True''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
