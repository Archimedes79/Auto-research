"""one-off: record the temporal-fusion ownership work and the nuScenes box test in plan.md 7.3"""
p = "plan.md"
s = open(p, encoding="utf-8").read()
anchor = "## 8. Semantic head"
assert s.count(anchor) == 1
note = (
    "### 7.4 Temporal fusion: owner review and repairs (2026-10-26, Mirko: this session owns the temporal fusion code)\n"
    "- Bug-hunter finding (the world memory never used in nuScenes training: no state carried, M = W = 0): the repair - clips of "
    "consecutive keyframes, the state carried with the ego step, detached per keyframe or kept (seq_bptt), warm / cold paired "
    "validation - is in the trainer session's UNCOMMITTED training/train.py (ClipDataset, run_clip); not duplicated, to be reviewed "
    "after its commit. t1 runs with seq_len = 1 are window-only (label them so).\n"
    "- Repaired here (e2b0ff5): the memory write had obs_mass = 1 everywhere, so the empty camera BEV outside the field of view "
    "overwrote what the memory had seen (exactly the cells the memory is for); now lift_pd.camera_visibility (cells whose height "
    "samples project into the image within the depth range) is the obs_mass of the camera branch; model self-test checks that the "
    "memory is written inside the view only. No effect on seq_len = 1 runs (memory empty there).\n"
    "- Open design point: the memory's static weight comes from TemporalFusion's OWN velocity head, which nothing supervises; in "
    "sequence training moving objects can be written into the static memory until it learns - use the supervised motion head "
    "(model.mover) or the GT box velocities for the static weight.\n"
    "- Real nuScenes boxes in the box test (tests/test_flow_explains.py --data nuscenes): the GT cache's keyframes k-1, k, k+1 "
    "(lanes = dividers, vehicles + two-wheelers, pedestrians), the true ego steps per gap, scene-wise split (1 in 5 held out), "
    "samples with a moving vehicle (about half of them); velocity error on moving vs static vehicles; --recursive scores at F1 "
    "the fresh estimate of (F1, F2), the (F0, F1) estimate carried to F1, and the carried estimate as the START of the (F1, F2) "
    "refinement (recursive speed refinement by comparing the next frame). Runs: baseline and the best synthetic setup, 2 seeds.\n\n")
s = s.replace(anchor, note + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
