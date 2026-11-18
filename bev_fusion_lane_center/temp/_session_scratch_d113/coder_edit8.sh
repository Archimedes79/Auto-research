set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for a, b in pairs:
        assert s.count(a) == 1, (p, s.count(a), a[:80])
        s = s.replace(a, b)
    open(p, "w", encoding="utf-8").write(s)
edit("training/train/jpast_train.py", [
 ("""        out_prev = out
    m.set_step(s0 + 1)
    return sum(losses)""", """        out_prev = out
    return sum(losses)"""),
 ("""the head's step counter is held at the value of
    the step for all its forwards and advanced once -> (loss, the list of the frames' info dicts)""", """every forward starts from the counter of the
    step (`set_step(s0)`), so the head's own increment leaves it advanced once after the last one -> (loss, the list of the frames' info dicts)"""),
])
edit("training/grid/test_jpast_train.py", [
 ("""    (T0, "the head's counter advances per frame", "    m.set_step(s0 + 1)\n    return sum(losses)", "    return sum(losses)"),
""", ""),
 ("""    (T0, "the switch is judged on the counter inside the step", "    learned = s0 >= JK.SWITCH", "    learned = m._step_py >= JK.SWITCH"),
""", ""),
 ("""    (T0, "arm z reads a past", "            out = m(fr)\n        else:", "            out = m(fr, {})\n        else:"),
""", ""),
 ("""    s_ = mk([[1, 1, 0], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)]], [0.0, 1.0], [4.0, 4.0], [0, 1], [False, True])""", """    s_ = mk([[1, 1, 1], [1, 0, 0]], [[(0, 1), (1, 2)], [(0, 1)]], [0.0, 1.0], [4.0, 4.0], [0, 1], [False, True])"""),
 ("""rep["against_off"] == dict(kept_gained_per_frame=0.5, kept_lost_per_frame=0.5, kept_flips_per_frame=1.0,""", """rep["against_off"] == dict(kept_gained_per_frame=0.5, kept_lost_per_frame=0.0, kept_flips_per_frame=0.5,"""),
])
print("ok")
EOF
