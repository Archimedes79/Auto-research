import types
import e346_lib as L, qformat, qgraph_eval as QE, qtargets as QT
stale = L.stale_qtargets()
fake = types.SimpleNamespace(**{k: v for k, v in vars(stale).items() if not k.startswith("__")})
for mod in (qformat, QE):
    keep = mod.QT
    for what, f in (("stale", stale), ("hidden", fake)):
        mod.QT = f
        try:
            with L.cell_size(1.0):
                try:
                    L.check_cell_size(1.0)
                    print(mod.__name__, what, "NOT CAUGHT")
                except AssertionError as e:
                    print(mod.__name__, what, "caught:", str(e)[:110])
        finally:
            mod.QT = keep
# toy node counts
g, _ = QE.toy_lane_gt()
for res in (2.0, 1.0):
    with L.cell_size(res):
        nt = qformat.oracle_prediction(g)["table"]
        print(res, len(nt["pos"]), nt["stats"])
