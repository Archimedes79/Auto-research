p = "evaluation/README.md"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep("python evaluation/bev_suite.py runs/t1_r4_soa time                                  # ms per frame, idle GPU only -> eval/latency.json\n",
    "python evaluation/bev_suite.py runs/t1_r4_soa time                                  # ms per frame, idle GPU only -> eval/latency.json\n"
    "python evaluation/bev_errors.py runs/t1_r4_soa --n 1000 [--check] [--class vehicle]    # every false positive and missed box of an evaluated run as a table (kind, frame, token, position, the command that redraws it) -> eval/errors_n1000.csv; no GPU, nothing re-run\n")
rep("`dashboard` | renders; error colours at the right cells |",
    "`dashboard`. Boxes on the prediction panels carry the errors in colour like the maps: right = amber (its GT box dashed), false = amber + green, missed = dashed white + blue, \"no care\" (less than half of the footprint in the observed area, D38) = violet, the GT boxes of 0.5 s earlier faded (moved into the present frame through the poses); the radar points are ONE layer (colour = Doppler direction: blue approaching, orange receding, grey standing; opacity = RCS, at least 50 %); velocity pictures use hue = direction, brightness = speed; `outputs` also shows the moving objects (> 0.5 m/s) and the stationary world as ground truth beside system, and the loss of the drivable-area / vehicle layers and the velocity error as a heat over the GT | renders; error colours at the right cells; the pure helpers (`radar_points`, `speed_dir_rgb`, `moving_layers`, `past_box_shapes`, `error_box_shapes`, `bce_map`, `loss_overlay`) are tested by `bev_views.selftest()` inside `bev_selftest.py` |\n"
    "| `bev_errors.py` | the error search: from the boxes `eval` stored (`scene_records_n<N>.pkl`) and the suite store's frame list it tells every false positive (duplicate / wrong class / isolated) and every missed box (below the score threshold / wrong class / missed / taken by a neighbour) with scene, frame, token, position, range, score or size and the command that redraws the frame; tables by class and kind, range, score and scene (the videos to watch); `--check` proves it against the card (with every detection counted, tp / fn / fp per class are the card's numbers: identical on `t22_B_s0`) | `--selftest` (every kind, counts equal the metric's matching on random worlds, the files, the scored window; in `tests/run_gates.sh`) |")
open(p, "w", encoding="utf-8").write(s)

p = "tests/run_gates.sh"
lines = open(p, encoding="utf-8").read().split("\n")
new = 'run "error search"       "$PY" evaluation/bev_errors.py --selftest       # FP / FN kinds, counts equal the metric\'s matching, frame trace through the files; no data'
if not any("bev_errors.py --selftest" in ln for ln in lines):
    i = next(k for k, ln in enumerate(lines) if "bev_timing.py --selftest" in ln)
    lines.insert(i + 1, new)
open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
