import os, time
import qgraph_view as QV
R = os.environ["BFLC_ROOT"]; out = os.path.join(R, "temp", "e273_view", "final")
t0 = time.time()
for sid, i in ((11, 12), (39, 20), (23, 25)):
    name, toks = QV.val_scenes()[sid]
    tok = toks[i]
    g = QV.load_gt(tok)
    for kind in ("oracle", "degraded"):
        QV.render_frame(tok, QV.make_pred(g, kind, tok), os.path.join(out, f"{name}_f{i:02d}_{kind}.png"), cams=True, g=g)
        print(name, i, kind, round(time.time() - t0, 1), "s", flush=True)
# the python route of a video: scene 39, 6 frames, a pred_fn callable (the network's route later), degraded
QV.render_scene(39, out, pred_fn=lambda t: QV.make_pred(QV.load_gt(t), "degraded", t), kind="degraded", video=True, frames=6, every=3)
QV.render_scene(11, out, kind="oracle", video=True, frames=6, every=3)
print("done", round(time.time() - t0, 1), "s")
