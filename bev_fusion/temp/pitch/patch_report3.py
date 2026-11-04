p = "../autoresearch/tools/run_report.py"
t = open(p, encoding="utf-8").read()
a = '''        open(os.path.join(d, "eval", "table.txt"), "w").write(f"seg_fov/seg/dt+0.00/walkway/all/all/iou@0   {val}\\nseg_fov/det/mATE   0.50\\n")'''
b = '''        open(os.path.join(d, "eval", "table.txt"), "w").write(f"seg_fov/seg/dt+0.00/walkway/all/all/iou@0   {val}\\nseg_fov/det/mATE   0.50\\n"
                                                              + "".join(f"seg_fov/extra/k{i}   {1.0 - i / 100:.3f}\\n" for i in range(60)))         # more than 40 rows: none may be cut'''
assert t.count(a) == 1, t.count(a)
t = t.replace(a, b)
a = '''    path, n = build(run, [ref], "selftest.run", "a note", 10, os.path.join(run, "report"))'''
b = '''    open(os.path.join(run, "eval", "table_train.txt"), "w").write("seg_fov/seg/dt+0.00/walkway/all/all/iou@0   0.50\\nseg_fov/det/mATE   0.40\\n"
                                                                  + "".join(f"seg_fov/extra/k{i}   {1.0 - i / 100 + 0.05:.3f}\\n" for i in range(60)))
    qd = os.path.join(run, "eval", "suite", "quant", "eval_n1")
    os.makedirs(qd)
    json.dump({"cost": {"ops": {"gmacs": 100.0, "method": "measured", "note": "test"}}}, open(os.path.join(qd, "manifest.json"), "w"))
    path, n = build(run, [ref], "selftest.run", "a note", 10, os.path.join(run, "report"))'''
assert t.count(a) == 1
t = t.replace(a, b)
a = '''    assert page.count("class='card'") == 1 and "exportReview" in page and "<select>" in page'''
b = '''    assert page.count("class='card'") == 1 and "exportReview" in page and "<select>" in page
    assert page.count("seg_fov/extra/k") >= 2 * 60 and "k59" in page, "the numbers and the gap table show every row, not the first 40"
    assert "Training against validation" in page and "train better" in page and "Compute (TOPS)" in page and "<b>2.00 TOPS</b>" in page'''
assert t.count(a) == 1
t = t.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("selftest extended")
