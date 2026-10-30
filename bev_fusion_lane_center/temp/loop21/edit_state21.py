def sub(p, pairs):
    t = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert a in t, (p, a[:60])
        t = t.replace(a, b)
    open(p, "w", encoding="utf-8").write(t)
sub("experiment_execution/PLAN.md", [
    ("| CPU 15 min | running |", "| CPU 15 min | done (E257): KEEP, vert100 15.0 -> 9.0 at eps 0.3 (Q -0.001), eps 0.2 holds all guards on both models (9.9 / 9.5); D20 filed |"),
    ("| CPU 30 min | planned |", "| CPU 30 min | done (E258): profile (relocation 21 %, anchoring 16 %); relocation stays (J -0.05); anchoring off holds on 4 of 4 j1rc seeds (dQ -0.0011 .. +0.0005, dJ -0.007 .. 0) -> registry `fast_recall_na`, saving ~1.3 - 2 ms estimated, idle repeat pending |"),
    ("| I3 | open slot | chosen from the results of I1 / I2 (candidates below) | card before the run | CPU | planned |", "| I3 | results video of the best system (Mirko's request) | - | no rule (presentation) | CPU | done: `reports/weekly/Lane_graph_recall_val.mp4` (recall, 2 median scenes, 80 frames) |"),
])
sub("reports/README.md", [
    ("| `Weeks1-2_LaneGraph_Presentation_EN.pptx` |", "| `Lane_graph_recall_val.mp4` (+ `_poster.png`) | results video of the `recall` system (VAL scenes 23 Boston and 7 Singapore One-North, the two whose scene Q is nearest the median; 80 frames, 3 fps): front camera, prediction red over ground truth green, per-frame F1@1.0; made by `temp/loop21/make_results_video.py` (mp4v: VLC / Windows Media Player) |\n| `Weeks1-2_LaneGraph_Presentation_EN.pptx` |"),
])
