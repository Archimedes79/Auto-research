"""One-off: corrections of experiments/t3_improve_loop/report.md after the fresh numbers review (2026-10-30)."""
p = "experiments/t3_improve_loop/report.md"
s = open(p, encoding="utf-8").read()
P = []


def rep(old, new):
    global s
    assert old in s, old[:60]
    s = s.replace(old, new, 1)


rep("# Eight tries to improve the camera + radar BEV system: what works, what does not, why, and where it stands against SOA",
    "# Eight tries to improve the camera + radar BEV system: what works, what does not, why, and how it relates to SOA")
rep("Status 2026-10-30 14:50. Eight tries done (plus one running). Every number comes from `runs/*/eval/metrics*.json` through `temp/study/collect.py`\n(`../t2_block_swaps/results/arms.md`); no number is typed by hand.",
    "Status 2026-10-30, reviewed once by a fresh reviewer against the primary files (corrections applied). Eight tries done (plus one running). The tables come from\n`runs/*/eval/metrics*.json` through `temp/study/collect.py` (`../t2_block_swaps/results/arms.md`); other numbers come from the named scripts and logs (probes `temp/calib/probe_*.log`,\nvelocity `temp/calib/velocity_baseline.py`, controls `try1_controls.md`, latency `runs/t1_r4_soa/eval/latency.json`); no number is typed by hand.")
rep("""1. **The reference system is robust against block swaps.** Of eleven exchanges, none is BETTER than the reference outside the noise band. Seven tie (gate fusion, CMX fusion,
   ConvGRU, pillars radar encoder, no temporal block, and the two camera-side fixes), four are worse (no depth weight in the lift, random-initialised BEV encoder, radar removed,
   camera removed). The best standard replacement is the pillars radar encoder (mAP2 0.349 against 0.339 and 0.324 for the two reference seeds: a candidate, not a result).""",
    """1. **No block swap shows an effect larger than the noise (about 0.05 mAP2 with one seed and 3000 steps).** Of eleven exchanges, none is BETTER than the reference. Seven are
   not distinguishable from their control (gate fusion, CMX fusion, ConvGRU, pillars radar encoder, no temporal block against the reference; the two camera-side fixes against
   camera-only), four are worse (no depth weight in the lift, random-initialised BEV encoder - both marginal, one metric just over the band -, radar removed, camera removed).
   The pillars radar encoder has the highest mAP2 (0.349 against 0.339 and 0.324 for the two reference seeds, a difference below the seed spread): worth a second seed, not a result.""")
rep("""2. **Both sensors are strong blocks and they complement each other.** radar only 0.062, camera only 0.210, camera + radar 0.339 / 0.324 (mAP2 = vehicle + pedestrian,
   four matching distances). The radar makes the camera precise (vehicle AP@0.5 0.045 -> 0.228), the camera makes the radar usable.""",
    """2. **The two sensors complement each other; the camera is the stronger one.** radar only 0.062 (pedestrian AP 0.000, vehicle AP@1 0.079), camera only 0.210, camera + radar 0.339 / 0.324
   (mAP2 = vehicle + pedestrian, four matching distances; 0.062 + 0.210 < 0.339). The radar makes the camera precise (vehicle AP@0.5 0.045 -> 0.228), the camera makes the radar usable.""")
rep("""3. **Weak spot 1: camera localisation.** Camera-only AP@0.5 is 0.045 while AP@2 is 0.448. Two camera-side fixes (trainable adapter on the frozen features, standard
   height-sum collapse of the lift) change nothing (0.211 and 0.205 against 0.210). Probes show the information is in the features and survives to the head input, so
   the limit is what stride-16 frozen image features with depth uncertainty can localise, not a broken path.""",
    """3. **Weak spot 1: camera localisation.** Camera-only vehicle AP@0.5 is 0.045 while vehicle AP@2 is 0.448. Two camera-side fixes (trainable adapter on the frozen features, standard
   height-sum collapse of the lift) do not change mAP2 beyond the band (0.211 and 0.205 against 0.210; individual metrics move by at most 0.02, section 3). Linear probes show that the
   vehicle / drivable / walkway information is in the lifted features and still linearly readable at the head input of the trained model (cell IoU, not 0.5 m localisation), which argues against a broken
   path; that the limit is stride-16 frozen features with depth uncertainty is the working hypothesis, not shown.""")
rep("""baseline exactly (2.39 m/s). After 7000 x 2 steps""", """baseline of its own matched vehicles exactly (2.39 m/s for the reference, 2.36-2.60 across arms). After 7000 x 2 steps""")
rep("""5. **Training length and data, not blocks, are the visible lever**: the same architecture trained 2.6 epochs (`t1_r4_soa`) reaches mAP2 0.364 (1000 frames) with lane dividers
   0.273 (0.181 at 3000 steps) and walkway 0.246 (0.191). Their runs are queued here (`t3_ref_long`, 700 scenes) but not finished.""",
    """5. **Longer training with a larger batch is the visible lever, not blocks**: the same architecture trained 7000 steps with gradient accumulation 2 (2.6 epochs, `t1_r4_soa`, a different
   schedule and config) reaches mAP2 0.364 on the same 1000 frames with lane dividers 0.273 (0.181 at 3000 steps) and walkway 0.246 (0.191); it also overfits more (train subset mAP2 0.462
   vs 0.361 on its full 3109-frame val, `metrics_train.json`). The amount of data was not varied; the clean tests (`t3_ref_long`, 700 scenes) are queued, not finished.""")
rep("""6. **Not comparable to SOA, and far from it**: ours is the front camera, 60 m, three class groups, short schedule; published R50 camera + radar systems reach 45-52 mAP
   (10 classes, six cameras, val). Section 8.""",
    """6. **Not comparable to SOA, and far from it**: ours is the front camera, 60 m, three class groups, short schedule; the published R50 camera + radar systems quoted in section 8
   report 45-52 mAP (10 classes, six cameras, val; secondary sources, unverified).""")
rep("""- Each arm changes ONE block (configs""", """- Each arm changes ONE block, except `fusion_cmx` which replaces the fusion AND the BEV encoder (CMX is its own pyramid + decoder) (configs""")
rep("""- Verdict rule (written before the runs): WORSE / BETTER = difference beyond the band (2 x seed spread of the stable metrics = 0.054) AND the paired scene bootstrap interval excludes 0;
  otherwise TIE (then the cheaper block). One seed per arm: a TIE means "not distinguishable from the reference here", not "equal".""",
    """- Verdict rule (written before the runs): WORSE / BETTER = difference beyond the band (2 x seed spread of the stable metrics = 0.054, set by walkway; the mAP2 seed spread is 0.015) AND the
  paired scene bootstrap interval excludes 0; otherwise TIE (then the cheaper block). One seed per arm and n = 2 seeds for the band: a TIE means "not distinguishable from the reference here",
  not "equal", and a WORSE based on one metric just over the band is marginal. The bootstrap resamples scenes, not training seeds: the two reference seeds themselves differ with `*` on walkway
  (-0.027*), pedestrian AP@1 (-0.020*) and AP@2 (-0.016*). Only effects above about 0.05 can be called with this design.""")
rep("drivable, walkway IoU, lane-divider IoU at 0.5 m tolerance.", "drivable and walkway IoU, lane-divider IoU at 0.5 m tolerance (the other maps at tolerance 0).")
rep("| sensors | radar only (`t3_radar_only`) | 0.062 | 0.079 | 0.000 | 0.586 | 0.112 | 0.098 | WORSE on everything |",
    "| sensors | radar only (`t3_radar_only`) | 0.062 | 0.079 | 0.000 | 0.586 | 0.112 | 0.098 | WORSE on every detection metric, drivable, walkway, divider |")
rep("| WORSE (vehicle AP@2, pedestrian AP@2) |", "| WORSE, marginal (vehicle AP@2 -0.057*, pedestrian AP@2 -0.056*; mAP2 -0.048 is inside the band; pedestrian AP@2 vs `ref_s1` -0.040 would be inside it) |")
rep("| WORSE (walkway -0.055*, divider -0.032*) |", "| WORSE, marginal (walkway -0.055*, one metric; vs `ref_s1` it would be -0.028 = the seed spread; also pedestrian AP@1 -0.024*, mAP2 -0.021 inside the band) |")
rep("| TIE, best (+0.011, pedestrian AP@1 +0.023*, divider +0.019*) |", "| TIE (highest mAP2, +0.011 is below the seed spread 0.015; pedestrian AP@1 +0.023* is as large as the seed pair's -0.020*) |")
rep("""Camera-only fixes are compared with camera-only `radar_off` (0.210), not with the reference: adapter `t3_cam_adapter` mAP2 0.211, vehicle AP@0.5 0.046 (against 0.045), divider 0.185 (0.164);
height-sum collapse `t3_cam_zsum` 0.205, vehicle AP@0.5 0.037, walkway 0.089 (0.107): both TIE.""",
    """Camera-only fixes are compared with camera-only `radar_off` (0.210), not with the reference (a comparison made with the same bootstrap tool outside `collect.py`; `collect.py` lists them against
`ref_s0`, where they are WORSE as camera-only systems): adapter `t3_cam_adapter` mAP2 0.211, vehicle AP@0.5 0.046 (against 0.045), divider 0.185 (0.164, +0.021* but inside the band);
height-sum collapse `t3_cam_zsum` 0.205, vehicle AP@0.5 0.037, walkway 0.089 (0.107, -0.018*), drivable 0.624 (0.637, -0.014*): both TIE by the rule, with small significant moves on single maps.""")
rep("| 2 | camera fix: standard height collapse | `t3_cam_zsum`: sum over heights, 1x1 (10 x fewer parameters) |", "| 2 | camera fix: standard height collapse | `t3_cam_zsum`: sum over heights, then 1x1 (5,120 instead of 40,960 weights in that 1x1) |")
rep("| maps +0.02, mAP +0.01 | 0.211 vs 0.210; vehicle AP@0.5 0.046 vs 0.045 | TIE, not adopted |", "| maps +0.02, mAP +0.01 | mAP2 0.211 vs 0.210 (miss); vehicle AP@0.5 0.046 vs 0.045; divider +0.021 (hit, one map) | TIE, not adopted |")
rep("shuffled radar gives 0.151 (mAP3); the long-trained system 0.031 / 0.036 and 0.156 (mAP3) |", "shuffled radar 0.151, empty radar 0.177 (all mAP3); the long-trained system 0.031 / 0.036 and 0.156 (mAP3) |")
rep("| radar only 0.062 | both sensors STRONG |", "| radar only 0.062, camera only 0.210, both 0.339 | radar alone is weak, each sensor is needed, the two complement each other |")
rep("| 0.349 | TIE, candidate |", "| 0.349 | TIE (highest mAP2, below the seed spread): needs a second seed |")
rep("""Camera-only AP rises from 0.045 at 0.5 m to 0.448 at 2 m:
  the camera finds vehicles but not their exact position, as expected from stride-16 features and a depth distribution of 0.5 m bins.""",
    """Camera-only vehicle AP rises from 0.045 at 0.5 m to 0.448 at 2 m (pedestrian 0.002 -> 0.104):
  the camera finds vehicles but not their exact position. The probes are cell IoU of vehicle / drivable / walkway masks (lane dividers: 0.031 against a prior of 0.018, not readable), so they show
  that coarse information survives, not 0.5 m localisation; stride-16 features and a depth distribution in 0.5 m bins are the working hypothesis for the precision limit.""")
rep("""In a 3000-step screening (about half an epoch of 428 scenes) the model is limited by training, not by the block design; every block can
  reach the same level.""", """In a 3000-step screening (about half an epoch: 9,000 clips of the 16,289 training samples of 428 scenes) these blocks reach the same level within the noise; I read that as training-limited,
  but the long runs that would show it are not finished.""")
rep("""The velocity loss weight (1.0) and the schedule leave the head at zero output; the zero-velocity baseline equals the measured error for `ref_s0` (2.39), `radar_off` (2.53),
  `t3_radar_only` (2.59).""", """The velocity loss weight (1.0) and the schedule leave the head at zero output: predicted speed 0.00 m/s, so the measured error equals the zero-velocity baseline of each model's matched vehicles
  (`ref_s0` 2.39, `radar_off` 2.53, `t3_radar_only` 2.59; 12 of the 13 3000-step models checked).""")
rep("""- **Encoder pre-training.** Random initialisation only hurts maps (walkway, divider), and CMX with a random-initialised encoder reaches the reference level: the pre-trained DualBEV encoder is less important
  for detection than assumed.""", """- **Encoder pre-training.** Random initialisation (`trunk_rand`) is worse on walkway (-0.055*), with smaller same-sign moves on pedestrian AP (-0.024*, -0.026*) and vehicle AP@2 (-0.014*); mAP2 -0.021 is
  inside the band. No arm isolates the encoder inside CMX (it changes fusion and encoder together), so nothing is concluded about the value of the pre-trained encoder for detection.""")
rep("system adds pedestrians and sharper maps.", "system detects more pedestrians (pedestrian AP@1 0.172 vs 0.141 on the same frames; one seed) and has sharper maps.")
rep("3000 optimiser steps on 428 scenes (half an epoch; the published systems train 20-24 epochs on 850 scenes)", "3000 optimiser steps on 428 scenes (half an epoch; the published systems train about 20-24 epochs on the 700 training scenes of the official split, six cameras)")
rep("""What carries over: radar adds to the camera
(published 5-10 mAP points on the same backbone; here camera 0.210 -> camera + radar 0.339 mAP2), a camera-only head localises coarsely, the standard fusion blocks are interchangeable. What is missing for a SOA claim:
six cameras, the full class set and NDS, 20 epochs, velocity learned, the full 850 scenes. No claim of SOA performance is made.""",
    """What carries over qualitatively: radar adds to the camera
(here camera 0.210 -> camera + radar 0.339 mAP2; the published camera-only versus camera + radar gaps were not verified for this report), a camera-only head localises coarsely, the standard fusion blocks are
interchangeable here. What is missing for a SOA claim: six cameras, the full class set and NDS, about 20 epochs, velocity learned, all 700 training scenes of the official split. No claim of SOA performance is made.""")
rep("3. A full C: disk (orphaned WSL swap files, cleaned with Mirko's OK) crashed `fusion_gate` at its last checkpoint and killed the queue for 4.5 h;",
    "3. A full C: disk (orphaned WSL swap files, cleaned with Mirko's OK) crashed `fusion_gate` at the write of its step-3000 checkpoint (last good checkpoint 2500) and killed the queue (about 4.5 h of idle GPU, per the experiment log);")
rep("1. Train longer and on more data (6000 steps queued; 700 scenes need the parts 7-10 caches, built, plus 11 GB of feature shards moved back from D:). Expected from the long run: dividers, walkway, velocity.",
    "1. Train longer and on more data (6000 steps queued; 700 scenes need the parts 7-10 caches, built; 11 GB of feature shards must move back from D:). Hypothesis to test: dividers, walkway, velocity improve.")
rep("3. Camera localisation: stride-8 features, depth refinement; the frozen detection features are not the limit (adapter TIE).", "3. Camera localisation: stride-8 features, depth refinement; the adapter on the frozen features did not help beyond the band.")
open(p, "w", encoding="utf-8", newline="\n").write(s)
q = "experiments/t3_improve_loop/soa_reference.md"
t = open(q, encoding="utf-8").read()
t = t.replace("3000-step schedule (about 0.55 epoch of the 428-scene training set; published systems train 20-24 epochs on 850 scenes with 6 cameras). The relation that carries over: in the\nliterature radar-camera fusion beats the camera-only counterpart by 5-10 mAP points on the same backbone; ours: camera 0.210 -> camera + radar 0.339 mAP2 (+0.13).",
              "3000-step schedule (about 0.55 epoch of the 428-scene training set; published systems train about 20-24 epochs on the 700 training scenes of the official split, 850 = train + val, with 6 cameras). Qualitatively: radar-camera\nfusion beats camera-only in the literature (the size of the gap on the same backbone is NOT verified here); ours: camera 0.210 -> camera + radar 0.339 mAP2 (+0.13).")
open(q, "w", encoding="utf-8", newline="\n").write(t)
print("ok")
