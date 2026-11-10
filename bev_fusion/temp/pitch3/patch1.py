def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)

patch("autoresearch/evalsuite/api.py", [
    ('''the evaluator code (the source's
        hash, and the adapter's own `evaluator_extra()` for the code that shapes numbers outside it)"""
        extra = self.ad.evaluator_extra() if hasattr(self.ad, "evaluator_extra") else None
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("split"), desc.get("set_hash"), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash"), extra])''',
     '''the evaluator code (the source's
        `evaluator_hash`: it must cover every file that shapes a number)"""
        return "e" + digest([rid, digest([[r.label, r.key] for r in keep], 40), desc.get("split"), desc.get("set_hash"), desc.get("gt_signature"), sorted(metrics), desc.get("evaluator_hash")])'''),
])
patch("autoresearch/evalsuite/README.md", [
    ("release inputs evaluator_extra evaluator_now", "release inputs evaluator_now"),
])
patch("bev_fusion/evaluation/bev_eval.py", [
    ('''def evaluator_extra() -> str:
    """the code that shapes the numbers outside the source's evaluator hash (bev_adapter.NUMBER_PATHS, which holds bev_views.py and bev_annotation_pts.py since D73): this adapter
    (the reference, the metrics without the loss) and the label-free path; it goes into the id of an evaluation, so an edit of either scores the stored predictions again
    instead of serving old numbers"""
    files = ("bev_eval.py", "bev_free.py")
    return hashlib.sha1("".join(_file_hash(os.path.join(ROOT, "evaluation", f)) for f in files).encode()).hexdigest()[:12]


''', ''),
])
patch("bev_fusion/evaluation/bev_adapter.py", [
    ('''`bev_views.py` also holds the page rendering, so a picture-only edit there changes the hash too, as it did in
# `bev_eval.evaluator_extra()` before). Not the selftests''', '''`bev_views.py` also holds the page rendering, so a picture-only edit there changes the hash too;
# `bev_eval.py` is the class interface's adapter: the reference and the metrics without the loss). Not the selftests'''),
    ('''("bev_adapter.py", "bev_task.py", "metrics.py", "bev_velocity.py", "bev_fusion_area.py", "bev_curves.py", "bev_views.py", "bev_annotation_pts.py")''',
     '''("bev_adapter.py", "bev_task.py", "metrics.py", "bev_velocity.py", "bev_fusion_area.py", "bev_curves.py", "bev_views.py", "bev_annotation_pts.py", "bev_eval.py")'''),
])
patch("bev_fusion/tests/test_evaluation_api.py", [
    ('''    assert len(bev_eval.evaluator_extra()) == 12, "the adapter's own code is part of the evaluation id"''',
     '''    assert os.path.join(ROOT, "evaluation", "bev_eval.py") in bev_adapter.NUMBER_PATHS, "the adapter's own code is part of the evaluator hash (and so of the evaluation id)"'''),
])
print("ok")
