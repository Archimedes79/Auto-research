"""E309 scratch: a 4-panel video (front camera | B | E312 | E287 p2) of whole VAL scenes (every 2nd keyframe), against the l6 GT with direction arrows.
usage: wsl bash tools/env/_py14.sh e309/render_scene.py <seed> <outdir rel to temp/e309> <scene> [<scene> ...]"""
import os, sys
src = open(os.path.join(os.environ["BFLC_ROOT"], "temp", "e309", "render.py"), encoding="utf-8").read().split("toks = []")[0]
argv = list(sys.argv)
sys.argv = [argv[0], argv[1], argv[2]]
exec(compile(src, "render.py", "exec"))
import cv2
for sc in argv[3:]:
    recs = [r for r in PF if str(r["scene"]) == sc][::2]
    d = os.path.join(out_dir, f"scene{sc}")
    os.makedirs(d, exist_ok=True)
    out_dir_save = out_dir
    out_dir = d
    paths = [render(r) for r in recs]
    out_dir = out_dir_save
    ims = [cv2.imread(p) for p in paths]
    H, W = ims[0].shape[:2]
    W2 = 1600
    H2 = int(round(H * W2 / W / 2) * 2)
    vp = os.path.join(out_dir, f"scene{sc}_B_E312_E287.webm")
    wr = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"vp09"), 2.0, (W2, H2))
    for im in ims:
        wr.write(cv2.resize(im, (W2, H2), interpolation=cv2.INTER_AREA))
    wr.release()
    print("video", vp, len(ims), "frames", flush=True)
