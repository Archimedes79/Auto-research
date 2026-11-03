import re
p = "evaluation/bev_suite.py"
t = open(p, encoding="utf-8").read()

# 1. curve headline rows in the printed / stored table
a = '    keys += [k for k in flat if k.startswith(("val_loss/", "ms/", "info/"))]'
b = '''    for layer in ("drivable_area", "walkway", "carpark_area", "ped_crossing", "lane_divider", "road_divider", "vehicle", "pedestrian"):      # --curves only
        keys += [f"curve/dt+0.00/{layer}/{m}" for m in ("ap", "roc_auc", "ece")]
    keys += [f"curve/dt+0.00/velocity/{sub}/{m}" for sub in ("obj", "moving") for m in ("epe", "rmse", "r2", "bias")]
    keys += [k for k in flat if k.startswith(("val_loss/", "ms/", "info/"))]'''
assert t.count(a) == 1
t = t.replace(a, b)

# 2. cmd_report forwards the two table paths of the gap block
a = '''    p.add_argument("--no-pictures", action="store_true")
    r = p.parse_args(rest)'''
b = '''    p.add_argument("--no-pictures", action="store_true")
    p.add_argument("--val-table", default="")
    p.add_argument("--train-table", default="")
    r = p.parse_args(rest)'''
assert t.count(a) == 1
t = t.replace(a, b)
a = '''(["--vs"] + r.vs if r.vs else []), env=env)'''
b = '''(["--vs"] + r.vs if r.vs else [])
                           + (["--val-table", r.val_table] if r.val_table else []) + (["--train-table", r.train_table] if r.train_table else []), env=env)'''
assert t.count(a) == 1
t = t.replace(a, b)

# 3. generalisation: never replaces stored numbers, refuses mismatching evaluations
i, j = t.index("def cmd_generalisation"), t.index("def main(argv=None)")
new = '''def cmd_generalisation(a, rest) -> int:
    """the same evaluation on the validation and on the TRAIN split (n frames each), then the report page with the train-vs-val table.
    n = the n of the stored plain val evaluation (its card), else the largest n in by_n/, else 600. A stored val evaluation of that n is reused; a val
    evaluation is only run when the run has none (it would replace the plain files, so a run with a val evaluation of ANOTHER n is refused: pass --n of it,
    or evaluate a copy of the run folder). Val and train must come from the same checkpoint step and the same GT signature, else nothing is compared."""
    import re
    p = argparse.ArgumentParser(prog="generalisation", allow_abbrev=False)
    p.add_argument("--n", type=int, default=0)
    r = p.parse_args(rest)
    if a.control:
        print("generalisation: not with --control (the file names carry the control; compare the plain runs)")
        return 2
    ev = os.path.join(a.run, "eval")
    by_n = os.path.join(ev, "by_n")
    have = sorted(int(m.group(1)) for f in (os.listdir(by_n) if os.path.isdir(by_n) else []) for m in [re.fullmatch(r"table_n(\d+)\.txt", f)] if m)
    plain = (_read_json(os.path.join(ev, "card.json")) or {}).get("frames")
    n = r.n or plain or (have[-1] if have else 600)
    paths = {s: os.path.join(by_n, f"table{tag}_n{n}.txt") for s, tag in (("val", ""), ("train", "_train"))}
    if not os.path.exists(paths["val"]) and plain and plain != n:
        print(f"generalisation: this run's stored val evaluation has {plain} frames; evaluating val with n={n} would replace its plain files. Use --n {plain} or a copy of the run folder.")
        return 2
    for split in ("val", "train"):
        if os.path.exists(paths[split]):
            print(f"generalisation: {split} n={n} already evaluated, reused", flush=True)
            continue
        b = argparse.Namespace(**vars(a))
        b.split = split
        code = cmd_eval(b, ["--n", str(n)])
        if code:
            return code
    mv, mt = (_read_json(os.path.join(by_n, f"metrics{tag}_n{n}.json")) or {} for tag in ("", "_train"))
    step = lambda m: (m.get("metrics") or {}).get("info/step")  # noqa: E731
    if mv.get("gt_signature") != mt.get("gt_signature") or step(mv) != step(mt):
        print(f"generalisation: val and train do not match (GT {mv.get('gt_signature')} / {mt.get('gt_signature')}, step {step(mv)} / {step(mt)}): not compared; "
              "evaluate the one that is out of date again")
        return 2
    print(f"generalisation: val and train, {n} frames each, step {step(mv)}, GT {mv.get('gt_signature')} (the TEST split is not touched: it needs a logged row under a recorded decision)")
    return cmd_report(a, ["--no-pictures", "--val-table", paths["val"], "--train-table", paths["train"]])


'''
t = t[:i] + new + t[j:]
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("bev_suite patched")
