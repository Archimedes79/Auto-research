def edit(p, pairs):
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (p, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


edit("evaluation/bev_views.py", [(
    '''    far = [x for x in shapes2 if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - 70) < 0.3][0]''',
    '''    # a ground-truth box just beyond the window (60.8 m) neutralises a detection inside it, as in the metric (the detection is ignored, not a false alarm)
    shapes3, cnt3 = error_box_shapes(dets + [box(59.4, 0, s=0.9)], gts + [box(60.8, 0)], shares + [1.0], area)
    assert cnt3 == dict(cnt, nocare=cnt["nocare"] + 1), (cnt3, cnt)
    far = [x for x in shapes2 if x.kind == "polygon" and abs(np.mean([q[1] for q in x.xy]) - 70) < 0.3][0]''')])

edit("evaluation/bev_selftest.py", [(
    '''    for v in ("overview", "outputs", "dashboard", "stages"):
        page = suite.qual.page(s, p, v, ctx)
        img = figure.render_page(page)
        assert page.panels and img.min() != img.max(), v''',
    '''    for v in ("overview", "outputs", "dashboard", "stages"):
        page = suite.qual.page(s, p, v, ctx)
        img = figure.render_page(page)
        assert page.panels and img.min() != img.max(), v
        gone = [pn.title for pn in page.panels if "not available" in str(pn.title)]
        assert not gone, (v, gone, "a decoration fell back: the new picture code failed silently on a frame that has everything it needs")'''),
    ('''def main() -> int:
    ap_golden, ap_real = "--golden" in sys.argv, "--real" in sys.argv''', '''def main() -> int:
    ap_golden, ap_real = "--golden" in sys.argv, "--real" in sys.argv
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None            # --only TEXT: just the checks whose name contains TEXT (a gate for one of them)'''),
    ('''        init_synthetic()
        todo = list(CHECKS)''', '''        init_synthetic()
        todo = [c for c in CHECKS if only is None or only in c[0]]
        assert todo, f"no check named like {only!r}"''')])

edit("tests/run_gates.sh", []) if False else None
print("ok")
