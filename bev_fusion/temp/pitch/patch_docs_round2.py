import sys

mode, path = sys.argv[1], sys.argv[2]
s = open(path, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


if mode == "interface":
    rep("| `compare(control, variant)` | `compare CONTROL VARIANT` | paired bootstrap, variant - control; refused on another GT or other frames |",
        "| `compare(a, b)` | `compare A B` | paired bootstrap, b - a (a = the control); refused on another GT or other frames |")
    rep("Ids: `<network>@r<hash>` the predictions, `<network>@e<hash>` a scored evaluation; the same question gives the same id. Frames: `val`, `small.val`, `spread:1000:large.val`, `random:3:7:val`,\n`scenes:a,b:val`, `frames:scene-0061/12:val` (the set is last).",
        "Ids: `<network>@r<hash>` the predictions, `<network>@e<hash>` a scored evaluation; the same question gives the same id. One network per process (a project's code modules are imported once):\nrun each network in its own process and compare the evaluations by their ids. Frames: `val`, `small.val`, `spread:1000:large.val`, `random:3[:SEED]:val`, `scenes:a,b:val`,\n`frames:scene-0061/12:val` (the set is last, and one the project defines).")
elif mode == "evalsuite":
    rep('''r.compare(ev.run("runs/t21_ctl_s0", "small.val", gt_key="v8"))      # variant - control, paired bootstrap''',
        '''ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids''')
elif mode == "bev":
    rep("python -m evalsuite Evaluation.compare CONTROL_ID VARIANT_ID                  # variant - control, refused on another GT or other frames",
        "python -m evalsuite Evaluation.compare A_ID B_ID                              # b - a (a = the control), refused on another GT or other frames; each network ran in its own process")
    rep("What differs from `bev_suite.py eval`: a network with a memory runs warm,",
        "One network per process: the code of a run (its snapshot) stays imported, `bev_eval.source` refuses a second snapshot in the same process. What differs from `bev_suite.py eval`: a network with a memory runs warm,")
open(path, "w", encoding="utf-8").write(s)
print("ok")
