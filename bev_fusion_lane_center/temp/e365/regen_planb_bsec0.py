"""scratch: store plain B's records (check-layer1, PASS only) and B' with the close-node step (bsec0_close, as `ctl --arms bsec0_close` writes its pickle) for both seeds, no card"""
import os, sys, pickle, types
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.join(os.environ["BFLC_ROOT"], "evaluation"))
import jtoken_eval as E
import qgraph_eval as QE
for seed in (0, 1):
    rc = E.check_layer1(types.SimpleNamespace(seed=seed, dump=None))
    assert rc == 0, rc
    frames = QE.val_frames()
    dump_path = E._abs(E.B_DUMPS["e292_a"].format(seed=seed))
    d = pickle.load(open(dump_path, "rb"))
    assert [y["row"] for y in d] == [f.row for f in frames]
    res = E.score_preds([E.pred_of_dump(y, E.CLOSE, True) for y in d], frames)
    E.show(f"bsec0_close e292_a s{seed}", res["graph"])
    pickle.dump(dict(res, dump=E._file_id(dump_path)), open(E.bsec0_close_path("e292_a", seed), "wb"))
