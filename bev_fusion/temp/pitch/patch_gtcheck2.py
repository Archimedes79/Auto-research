p = "evaluation/bev_gtcheck.py"
t = open(p, encoding="utf-8").read()


def rep(a, b):
    global t
    assert t.count(a) == 1, (a[:70], t.count(a))
    t = t.replace(a, b)


rep('''            side = s[dt]["meta"]
            if side.get("steps"):
                R, t = to_frame(s[0.0], s[dt])
                d_pose = float(np.hypot(*(-R.T @ t))) if dt > 0 else float(np.hypot(*t))
                dev.append(abs(d_pose - float(np.hypot(*steps_displacement(side["steps"])[:2]))))
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}, (float(np.mean(dev)) if dev else float("nan"))''',
    '''            side = s[dt]["meta"]
            if side.get("steps"):                                       # steps: the ego motion from cur to next, from prev to cur (what the time mover is given)
                a, b = (s[0.0], s[dt]) if dt > 0 else (s[dt], s[0.0])
                px, py, pyaw = rel_pose(a["meta"]["T_e2g"], b["meta"]["T_e2g"])
                sx, sy, syaw = steps_displacement(side["steps"])
                dev.append((float(np.hypot(px - sx, py - sy)), abs(math.remainder(pyaw - syaw, 2 * math.pi))))
    mean_xy = float(np.mean([d[0] for d in dev])) if dev else float("nan")
    return {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in acc.items()}, {"xy_mean": mean_xy, "xy_max": max((d[0] for d in dev), default=float("nan")),
                                                                                   "yaw_max": max((d[1] for d in dev), default=float("nan"))}''')
rep('''def steps_displacement(steps):''', '''def rel_pose(Ta, Tb):
    """the pose of ego b in the ego frame of a (x, y, yaw) from the 4x4 poses ego -> global"""
    T = np.linalg.inv(np.asarray(Ta, np.float64)) @ np.asarray(Tb, np.float64)
    return float(T[0, 3]), float(T[1, 3]), float(math.atan2(T[1, 0], T[0, 0]))


def steps_displacement(steps):''')
rep("import argparse\nimport glob\nimport os\nimport sys", "import argparse\nimport glob\nimport math\nimport os\nimport sys")
rep('''    assert dev < 0.2, f"the displacement of the poses and of meta['steps'] differ by {dev:.3f} m"''',
    '''    assert dev["xy_max"] < 0.15 and dev["yaw_max"] < 0.01, f"the ego motion of the poses and of meta['steps'] differ: {dev}"       # measured: 0.07 m, 9e-5 rad''')
rep('''    print(f"  displacement of the recorded poses against the one integrated from meta['steps']: mean abs difference {dev:.3f} m")''',
    '''    print(f"  ego motion of the recorded poses against the one integrated from meta['steps'] (x, y, yaw): xy mean {dev['xy_mean']:.3f} m, max {dev['xy_max']:.3f} m, yaw max {dev['yaw_max']:.5f} rad")''')
# the optional clause of the road divider floor: plain and strict
rep('''            assert iou[(layer, dt, "warped")] > max(floor, iou[(layer, dt, "unwarped")] + 0.1) - 1e-9 or layer == "road_divider" and iou[(layer, dt, "warped")] > 2 * iou[(layer, dt, "unwarped")], (layer, dt, iou)''',
    '''            assert iou[(layer, dt, "warped")] > max(floor, iou[(layer, dt, "unwarped")] + 0.1), (layer, dt, iou)          # thin lines: 0.61 - 0.66 measured, floor 0.5''')
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched")
