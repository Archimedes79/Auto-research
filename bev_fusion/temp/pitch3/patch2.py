def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)

patch("autoresearch/evalsuite/api.py", [
    ('''            desc = source.describe()
            scenes, keep = fr.resolve(source)
            res["gt"]''', '''            desc = source.describe()
            _, keep = fr.resolve(source)
            res["gt"]'''),
    ('''            desc = source.describe()
            scenes, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)''', '''            desc = source.describe()
            _, keep = fr.resolve(source)
            self._need_gt(desc, gt_key)'''),
    ('''        info, store, q = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict"''', '''        info, store, _ = self._open(id)
        rs = self._predictions(info)
        return {"id": id, "predict"'''),
    ('''        info, store, q = self._open(id)
        return q.report(store, render=0)''', '''        _, store, q = self._open(id)
        return q.report(store, render=0)'''),
    ('''        info, store, q = self._open(id)
        self._same_network(info, self._predictions(info))
        fr = Frames.parse(info["frames"])''', '''        info = self._open(id)[0]
        self._same_network(info, self._predictions(info))
        fr = Frames.parse(info["frames"])'''),
])
patch("autoresearch/evalsuite/check.py", [
    ("from .contracts import FrameRef, Page, Prediction, Sample", "from .contracts import Page, Prediction, Sample"),
    ("    from .qualitative import DebugView, Qualitative\n", "    from .qualitative import DebugView\n"),
])
patch("autoresearch/evalsuite/contracts.py", [
    ("from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple", "from typing import Any, Dict, List, Optional, Sequence, Tuple"),
])
patch("autoresearch/evalsuite/qualitative.py", [
    ("from .contracts import Flag, FrameRef, Page, Panel, Prediction, Sample, Source, Subject, Task, View, ViewContext", "from .contracts import Flag, Page, Prediction, Sample, View, ViewContext"),
])
patch("autoresearch/evalsuite/runner.py", [
    ("from typing import Iterator, List, Optional, Sequence, Tuple", "from typing import Iterator, Optional, Sequence, Tuple"),
])
patch("bev_fusion/evaluation/bev_free.py", [
    ('''        b, ns = self.b, self.b.ns
        T, model, cfg, device = ns.T, b.model, b.cfg, b.device
        names = set(taps)''', '''        b, ns = self.b, self.b.ns
        T, model, device = ns.T, b.model, b.device
        names = set(taps)'''),
])
patch("bev_fusion/evaluation/bev_modules.py", [
    ("    ns, model, cfg = subj.ns, subj.model, subj.cfg\n    T, dev = ns.T, subj.device\n    for k, (scene, idx) in enumerate(refs):",
     "    ns, model = subj.ns, subj.model\n    T, dev = ns.T, subj.device\n    for k, (scene, idx) in enumerate(refs):"),
])
patch("bev_fusion/evaluation/bev_task.py", [
    ("from evaluation.metrics import DetMeter, SegMeter, bin_label, dt_key, in_front", "from evaluation.metrics import DetMeter, SegMeter, dt_key, in_front"),
])
patch("bev_fusion/evaluation/bev_views.py", [
    ("import math\nimport textwrap\nfrom typing", "import math\nfrom typing"),
    ('''            sp = np.hypot(o0["vhat"][0], o0["vhat"][1])
''', ''),
])
patch("bev_fusion/evaluation/stream_drift.py", [
    ("        miss, unexp = m.load_state_dict(", "        miss, _ = m.load_state_dict("),
])
patch("bev_fusion/evaluation/stream_timing.py", [
    ("import argparse\nimport dataclasses\nimport json", "import argparse\nimport json"),
])
print("ok")
