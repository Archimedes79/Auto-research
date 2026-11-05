"""t21 scratch: writes the four configs of training/configs/ from t20_base.json (a new file each, never an edit of a stored one)."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CFG = os.path.join(ROOT, "training", "configs")
base = json.load(open(os.path.join(CFG, "t20_base.json")))

COMMON = ("t21 (D51): t20_base.json (the t20 network: dual camera features, L4 lane branch, motion block, delta slots, 1 m centre-line head, hygiene "
          "fixes, split 'large') warm-started from the finished baseline runs/t20_base_s0 (init_from: its EMA weights, optimiser and schedule fresh; "
          "cam_buffer adds no parameter, so no key is fresh) and trained 2000 steps at lr 1e-4 with warm-up 100 (t20_base: 6000 steps, lr 3e-4); "
          "everything else as t20_base.json (seed 0, batch 3, accum 2, clips of 3 keyframes with the state kept in the graph, flip augmentation). "
          "The arms of the experiment (pair, pair4, rec, ctl) differ in model.cam_buffer, n_cam and cam_stride only: ")
ARMS = {
    "t21_pair": ("pair", 2, 2, "model.cam_buffer 'pair', n_cam 2: the camera temporal block's window is ONE buffered frame, the previous one "
                 "(frames t and t-2 sweeps = 0.167 s apart; in stream mode the previous raw camera BEV comes from state['buf']). The motion block "
                 "reads the same two raw BEVs, so its time baseline is 0.167 s instead of the window's oldest frame (0.333 s)."),
    "t21_pair4": ("pair", 2, 4, "like t21_pair with cam_stride 4: the buffered frame is 0.333 s earlier, the motion block's baseline of t20_base."),
    "t21_rec": ("recurrent", 3, 2, "model.cam_buffer 'recurrent', n_cam 3: the temporal block's own output of the previous frame is the buffer "
                "(each frame is lifted once and fused with the previous fused state, moved by the block's splat and velocity head); a batch unrolls "
                "its three frames oldest to newest and continues from the previous keyframe's buffer (the frame chain at stride 2 is gapless), so the "
                "training chain is up to 9 frames deep while a stream is unbounded. The motion block reads the current and the previous raw BEV "
                "(0.167 s)."),
    "t21_ctl": ("window", 3, 2, "model.cam_buffer 'window' (the t20_base code path, bit for bit), n_cam 3: the control, the same continued training "
                "on the window of three frames."),
}
for name, (buf, n_cam, stride, text) in ARMS.items():
    c = json.loads(json.dumps(base))
    c["_about"] = COMMON + text
    c["name"] = name
    c.update(steps=2000, lr=0.0001, warmup=100, eval_every=1000, save_every=500, n_cam=n_cam, cam_stride=stride,
             init_from="runs/t20_base_s0/ckpt_00006000.pt")
    c["model"]["cam_buffer"] = buf
    with open(os.path.join(CFG, name + ".json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(c, fh, indent=1)
        fh.write("\n")
    print("wrote", name)
