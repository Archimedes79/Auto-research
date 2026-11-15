"""scratch: write the four t25 configs from t21_pair4.json (only name, _about and the listed keys change); first check that the
json round trip of t21_pair4.json is byte-identical, so the files differ from it by those keys only"""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
CFG = os.path.join(ROOT, "training", "configs")
src_text = open(os.path.join(CFG, "t21_pair4.json"), encoding="utf-8", newline="").read()
src = json.loads(src_text)
dump = lambda d: json.dumps(d, indent=1, ensure_ascii=False)               # noqa: E731
print("round trip identical:", dump(src) == src_text.rstrip("\n"), "| trailing newline in source:", src_text.endswith("\n"))

COMMON = ("t25 (card experiments/t25_one_frame_one_scan/, D91): t21_pair4.json (the t21 pair-P4 recipe: warm start runs/t20_base_s0, 1500 steps, lr 1e-4, warm-up 100, ema 0.99, clips of 3 "
          "keyframes, GT v13 / lane l6 as the t20 chain) with batch_size 2 and accum 3 (the same 6 clips per step as batch 3 and accum 2; the recursive arms do not fit at batch 3, "
          "t21_rec.json) and ")
REC = ("n_cam 3 and cam_stride 2 (the t21_rec.json frame chain, 6 Hz) and model.cam_buffer 'recurrent' (each frame lifted once and fused with the previous fused state; "
       "no new camera parameter, the warm start loads exactly)")
ABOUT = {
    "t25_ctl": COMMON + "no other change: the matched control of the card, retrained on the current GT (the stored t21.pair4_s0 was made on GT v9). cam_buffer 'pair' (n_cam 2, cam_stride 4: a "
                        "pair batch takes two frames), 0.5 s of radar sweeps accumulated at point level, no radar memory.",
    "t25_rec_r": COMMON + REC + ". Arm R' of the card (attribution): the recursive camera with the unchanged 0.5 s radar window (radar_window 0.5), no radar memory (rad_temporal none).",
    "t25_rec_scan": COMMON + REC + " and radar_window 0.09 (ONE radar sweep: the newest sweep of each of the five radars, about 54 points in the grid per keyframe against 300 for 0.5 s). "
                                   "Arm A1 of the card: what one sweep costs, no radar memory (rad_temporal none).",
    "t25_rec_scan_mem": COMMON + REC + " and radar_window 0.09 as t25_rec_scan.json, plus the radar recursion: model.radar_per_frame (one radar set per camera frame, batch['radar_seq']), "
                                       "model.rad_temporal 'window_memory' with model.rad_static_gate (the memory is written only where the measured Doppler calls the cell static). Arm A2 of "
                                       "the card: the past of the radar lives in the feature state, stepped with the camera frames and the ego motion; it enters as a residual that starts as "
                                       "the identity (radar features + rad_mix(block output), rad_mix zero-initialised), so at step 0 the network equals t25_rec_scan.json; the keys "
                                       "rad_temporal.* and rad_mix.* are fresh at the warm start.",
}
CHANGE = {"t25_ctl": ({"batch_size": 2, "accum": 3}, {}),
          "t25_rec_r": ({"batch_size": 2, "accum": 3, "n_cam": 3, "cam_stride": 2}, {"cam_buffer": "recurrent"}),
          "t25_rec_scan": ({"batch_size": 2, "accum": 3, "n_cam": 3, "cam_stride": 2, "radar_window": 0.09}, {"cam_buffer": "recurrent"}),
          "t25_rec_scan_mem": ({"batch_size": 2, "accum": 3, "n_cam": 3, "cam_stride": 2, "radar_window": 0.09},
                               {"cam_buffer": "recurrent", "radar_per_frame": True, "rad_temporal": "window_memory", "rad_static_gate": True})}
for name, (top, model) in CHANGE.items():
    d = json.loads(src_text)
    d["_about"], d["name"] = ABOUT[name], name
    d.update(top)
    d["model"].update(model)
    out = os.path.join(CFG, name + ".json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write(dump(d) + ("\n" if src_text.endswith("\n") else ""))
    print("wrote", out)
