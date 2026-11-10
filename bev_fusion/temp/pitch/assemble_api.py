old = open("evalsuite/api.py", encoding="utf-8").read()
a = old.index("# ------------------------------------------------------------------------------------------------ frames")
b = old.index("# ------------------------------------------------------------------------------------------------ the adapter")
frames = old[a:b]
head = open("../bev_fusion/temp/pitch/api_head.py", encoding="utf-8").read()
tail = open("../bev_fusion/temp/pitch/api_tail.py", encoding="utf-8").read()
new = head + frames + tail
new = new.replace('''            if self._need_gt(desc, gt_key) and any(not rs.has(r) for r in keep):
                raise EvaluationError(f"{sum(not rs.has(r) for r in keep)} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0.0, False, progress)''',
'''            self._need_gt(desc, gt_key)
            missing = sum(not rs.has(r) for r in keep)
            if missing:
                raise EvaluationError(f"{missing} of the {len(keep)} frames are not in {results}: `predict` them first, or use `run`")
            return self._score(net, source, desc, rs, keep, fr, metrics, 0.0, 0, True, progress)''')
open("evalsuite/api.py", "w", encoding="utf-8").write(new)
print(len(new.splitlines()), "lines")
