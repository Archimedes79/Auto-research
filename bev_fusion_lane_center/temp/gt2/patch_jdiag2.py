p = "evaluation/jdiag.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub("    wsl bash tools/env/_py14.sh jdiag.py selftest\n", "    wsl bash tools/env/_py14.sh jdiag.py snaprule e287_p2 e305_e                    the rule of the E311 card on runs/<net>_s<S>/ends_card_snap_thr<t>.json (`jends_eval.py score --snap --thr t`): topo F1 over the plain graph > +0.02 with F1@1.0 >= -0.01, dup_frac not higher, vertices <= +1.0, junction F1 >= -0.03; both seeds of every net\n    wsl bash tools/env/_py14.sh jdiag.py selftest\n")
sub("def selftest():", '''def rule311(diff):
    """the rule of the E311 card applied to the differences (head - plain graph) of one scored read-out"""
    return diff["topo_F1"] > 0.02 and diff["F1@1.0"] >= -0.01 and diff["dup_frac"] <= 0 and diff["vert100_dp"] <= 1.0 and diff["junction_F1"] >= -0.03


def snaprule(nets, thrs=(0.5, 0.2, 0.1)):
    cells = {}
    for net in nets:
        for thr in thrs:
            for s in (0, 1):
                p = os.path.join(ROOT, "runs", f"{net}_s{s}", f"ends_card_snap_thr{thr}.json")
                if not os.path.exists(p):
                    continue
                d = json.load(open(p))["difference"]
                cells[(net, thr, s)] = (rule311(d), d)
                print(f"{net:9s} thr {thr} s{s}: {'HOLDS' if cells[(net, thr, s)][0] else 'fails'} topo {d['topo_F1']:+.4f} F1@1.0 {d['F1@1.0']:+.4f} dup {d['dup_frac']:+.4f} vert {d['vert100_dp']:+.2f} jF1 {d['junction_F1']:+.4f}")
    for thr in thrs:
        got = [cells.get((net, thr, s)) for net in nets for s in (0, 1)]
        print(f"thr {thr}: {sum(1 for c in got if c)}/{len(got)} cells scored; all hold: {all(c and c[0] for c in got)}")


def selftest():''')
sub('    m = maps_summary(d)', '''    ok = dict(topo_F1=0.025, **{"F1@1.0": 0.01}, dup_frac=-0.03, vert100_dp=0.2, junction_F1=0.02)
    assert rule311(ok) and not rule311(dict(ok, topo_F1=0.02)) and not rule311(dict(ok, dup_frac=0.001)) and not rule311(dict(ok, vert100_dp=1.1)) and not rule311(dict(ok, junction_F1=-0.031)) and not rule311(dict(ok, **{"F1@1.0": -0.011}))
    m = maps_summary(d)''')
sub('    elif cmd == "maps" and len(sys.argv) == 4:', '    elif cmd == "snaprule" and len(sys.argv) > 2:\n        snaprule(sys.argv[2:])\n    elif cmd == "maps" and len(sys.argv) == 4:')
open(p, "w", encoding="utf-8").write(s)
print("ok")
