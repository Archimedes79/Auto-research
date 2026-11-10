def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)
patch("bev_fusion/evaluation/bev_legacy.py", [
    ('''def det_ap(frames_by_scene, scenes, cls, thr=2.0):
    frames = [([d for d in ds if d["group"] == cls], [g for g in gs if g["group"] == cls])
              for s in scenes for ds, gs in frames_by_scene[s]]
    frames = [([d for d in ds if in_front(d)], [g for g in gs if in_front(g)]) for ds, gs in frames]
    m, n_gt = match(frames, thr)
    return ap_nuscenes([x[1] for x in m], n_gt)


''', ''),
    ("(equal to re-running det_ap, but fast)", "(equal to matching all scenes in one pass, but fast: bev_selftest.former_det_ap is that pass)"),
])
patch("bev_fusion/evaluation/bev_selftest.py", [
    ("from evaluation.metrics import DetMeter, SegMeter, dt_key, flatten  # noqa: E402\n", "from evaluation.metrics import DetMeter, SegMeter, ap_nuscenes, dt_key, flatten, in_front  # noqa: E402\n"),
    ('''def same(a, b, tol=1e-12):''', '''def former_det_ap(frames_by_scene, scenes, cls, thr=2.0):
    """the AP of the former evaluate_run: ALL listed scenes matched in one pass (the greedy matching of a detection only involves its own frame); the reference of bev_legacy's per-scene flags"""
    frames = [([d for d in ds if d["group"] == cls and in_front(d)], [g for g in gs if g["group"] == cls and in_front(g)])
              for s in scenes for ds, gs in frames_by_scene[s]]
    m, n_gt = metrics_match(frames, thr)
    return ap_nuscenes([x[1] for x in m], n_gt)


def same(a, b, tol=1e-12):'''),
    ("legacy_ap = bev_legacy.det_ap(frames_by_scene", "legacy_ap = former_det_ap(frames_by_scene"),
    ("the former global greedy matching (legacy det_ap) on the same boxes", "the former global greedy matching (former_det_ap) on the same boxes"),
])
print("ok")
