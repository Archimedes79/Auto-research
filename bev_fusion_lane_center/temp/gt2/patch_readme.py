def insert_after(path, startswith_marker, rows):
    lines = open(path, encoding="utf-8").read().split("\n")
    idx = max(i for i, l in enumerate(lines) if l.startswith(startswith_marker))
    for r in reversed(rows):
        lines.insert(idx + 1, r)
    open(path, "w", encoding="utf-8").write("\n".join(lines))


insert_after("training/grid/README.md", "| `qendnet.py`", [
    "| `jgt.py` | E285 (D82): the junction GT layer made on the lane side from the lane GT: junction zone (the connectors grown by 1.8 m), conflict core (where connectors of different pairs cross / merge / diverge), arms with lane counts, entry / exit ranks, the order-preserving pairs and their count (`order`), the 1 m maps (`fields`). Test: `python jgt.py`. |",
    "| `qjunction.py` | E285: the targets of the junction maps (zone + core, entry / exit heat with heading, stop-line / crossing masks at 0.5 m) from `jgt.py` and the paint sidecar of `data/prep/junction_paint.py`. Test: `python qjunction.py`. |",
    "| `jmaps.py` | E286 (D82): the junction maps head (`JMaps`, ctx none / dil / attn) on the pyramid's features, `JModel` (pyramid from a baseline run + the head), the soft loss (BCE + dice, cosine heading) and the keypoint read-out (3 x 3 peaks, the one hard step, named). Test: `python jmaps.py`. |",
])
insert_after("evaluation/README.md", "| `qends_eval.py`", [
    "| `jmaps_eval.py` | E286: the junction maps head on the 873 VAL frames: zone / core IoU, stop-line / crossing precision and recall, entry / exit keypoint recall and precision (3 x 3 peaks, 1.5 / 3 m, 30 degrees) next to the baseline's own event heat maps, heading error. `score --run <run>`, `selftest`. |",
])
insert_after("tools/README.md", "| `viz/qgraph_view.py`", [
    "| `viz/junction_view.py` | E285 / E286: pictures of the junction GT layer (zone, conflict core, stop lines, crossings, ends with arm.rank labels, pairs, order rule) and, with `--pred <run>`, of the junction maps head (heat, keypoints, zone / core, stop lines). `frame <token or scene-0221:12> --out DIR`, `--selftest`. |",
])
print("ok")
