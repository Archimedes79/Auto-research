"""The pictures and videos of a trained lane head (E273) for the result folder: `results/<run id>/{figures,videos}/` (next to the package.json of `package.py`) (not in git; the package names the command).

    wsl bash tools/env/_py14.sh qgraph_results.py --run e273_p_s0 [--scenes 11 39 23] [--every 1] [--map error|exact] [--device cuda]

Scenes = VAL scene ids of `qgraph_view.py list` (11 = scene-0221 and 39 = scene-0108: the two junction scenes, 23 = scene-0092: a straight road; the VAL split has no night scene). Every keyframe of a scene
is rendered with the head's real prediction (`qpredict.make_pred_fn`: no ground truth in the forward path) by `qgraph_view.render_scene`; one mp4 per scene. A frame picture for the key frames is
copied to `figures/`. CPU is enough (`--device cpu`, slower).
"""
import argparse
import os
import shutil
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub in ("tools/viz", "training/grid", "training/data", "training", "evaluation", ".."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub)))
import qgraph_view as QV  # noqa: E402
import qpredict as QP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--id", default=None, help="the run id of the launcher (E273.p_s0): the result folder is results/<id>/")
    ap.add_argument("--scenes", nargs="*", default=["11", "39", "23"])
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--map", default="error")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", a.id or a.run)
    vid, fig = os.path.join(out, "videos"), os.path.join(out, "figures")
    os.makedirs(vid, exist_ok=True)
    os.makedirs(fig, exist_ok=True)
    fn = QP.make_pred_fn(a.run, a.map, a.device, keep_maps=True)
    for sc in a.scenes:
        sc = int(sc) if sc.isdigit() else sc
        tmp = os.path.join(out, "frames", str(sc))
        os.makedirs(tmp, exist_ok=True)
        QV.render_scene(sc, tmp, pred_fn=fn, kind=a.run, video=True, every=a.every)
        for f in os.listdir(tmp):
            if f.endswith(".mp4"):
                shutil.move(os.path.join(tmp, f), os.path.join(vid, f))
        pngs = sorted(f for f in os.listdir(tmp) if f.endswith(".png"))
        if pngs:
            shutil.copy(os.path.join(tmp, pngs[len(pngs) // 2]), os.path.join(fig, pngs[len(pngs) // 2]))
    print("videos:", sorted(os.listdir(vid)), "| figures:", sorted(os.listdir(fig)), "| all frames:", os.path.join(out, "frames"))


if __name__ == "__main__":
    main()
