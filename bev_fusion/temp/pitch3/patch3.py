def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:100])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8", newline="\n").write(s)

NEW_FN = '''def outputs_of(model, res, ns, refine):
    """the outputs of every dt of one model result (sample 0) the way both Subjects return them (BEVSubject here, bev_free.FreeSubject): maps as probabilities and distances, the decoded
    boxes (`dets`, and `detr` when the run refines), the velocity maps, and with a 1 m centre-line head its layers and direction"""
    grid, spec = model.grid, model.spec
    refined = ({dt: model.boxes(res, dt, k=100, threshold=0.05, rescore=True) for dt in res["out"]} if refine else {})
    dts = {}
    for dt, p in res["out"].items():
        prob, dist = predictions(p, 0, spec)
        dts[dt] = {"prob": prob, "dist": dist,
                   "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
                   "dets": ns.decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, grid, ns.NUSC_GROUPS, k=100, threshold=0.05)[0],
                   "detr": list(refined[dt][0]) if refine else None,
                   "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
                   "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}     # the object head's raw velocity map (bev_velocity centres)
        if getattr(model, "spec_1m", None):                                   # the 1 m centre-line head: its presence is a layer of `prob`, its direction goes to the angle check
            p1, dts[dt]["dir"] = predictions_1m(p, 0, model.spec_1m)
            prob.update(p1)
    return dts


'''
OLD_BLOCK = '''                grid, spec = self.model.grid, self.model.spec
                refined = ({dt: self.model.boxes(res, dt, k=100, threshold=0.05, rescore=True) for dt in res["out"]} if self.refine else {})
                dts = {}
                for dt, p in res["out"].items():
                    prob, dist = predictions(p, 0, spec)
                    dts[dt] = {"prob": prob, "dist": dist,
                               "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
                               "dets": ns.decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, grid, ns.NUSC_GROUPS, k=100, threshold=0.05)[0],
                               "detr": list(refined[dt][0]) if self.refine else None,
                               "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
                               "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}     # the object head's raw velocity map (bev_velocity centres)
                    if getattr(self.model, "spec_1m", None):                          # the 1 m centre-line head: its presence is a layer of `prob`, its direction goes to the angle check
                        p1, dts[dt]["dir"] = predictions_1m(p, 0, self.model.spec_1m)
                        prob.update(p1)
'''
patch("bev_fusion/evaluation/bev_adapter.py", [
    ("def predictions_1m(p, b, spec_1m):", NEW_FN + "def predictions_1m(p, b, spec_1m):") if False else ("\n\ndef predictions_1m(p, b, spec_1m):", "\n\n" + "@@OUTPUTS@@" + "def predictions_1m(p, b, spec_1m):"),
    (OLD_BLOCK, "                dts = outputs_of(self.model, res, ns, self.refine)\n"),
])
# the new function goes after predictions_1m (it calls it at run time only, so the order does not matter; keep the two together)
s = open("bev_fusion/evaluation/bev_adapter.py", encoding="utf-8").read()
s = s.replace("@@OUTPUTS@@", "")
marker = "    return prob, direction\n"
assert s.count(marker) == 1
s = s.replace(marker, marker + "\n\n" + NEW_FN.rstrip("\n") + "\n", 1)
open("bev_fusion/evaluation/bev_adapter.py", "w", encoding="utf-8", newline="\n").write(s)

patch("bev_fusion/evaluation/bev_free.py", [
    ("                outputs = self._outputs(res)", "                outputs = A.outputs_of(b.model, res, ns, b.refine)"),
    ('''
    def _outputs(self, res) -> Dict[float, Dict[str, Any]]:
        """the outputs of every dt, as BEVSubject.predict builds them (copied from it, see the head of this file)"""
        b, ns = self.b, self.b.ns
        model = b.model
        grid, spec = model.grid, model.spec
        refined = ({dt: model.boxes(res, dt, k=100, threshold=0.05, rescore=True) for dt in res["out"]} if b.refine else {})
        dts = {}
        for dt, p in res["out"].items():
            prob, dist = A.predictions(p, 0, spec)
            dts[dt] = {"prob": prob, "dist": dist,
                       "mass": p["mass"][0, 0].float().cpu().numpy() if p.get("mass") is not None else None,
                       "dets": ns.decode({k: v[:1].float().cpu() for k, v in p["obj"].items()}, grid, ns.NUSC_GROUPS, k=100, threshold=0.05)[0],
                       "detr": list(refined[dt][0]) if b.refine else None,
                       "vhat": p["v_hat"][0].float().cpu().numpy() if p.get("v_hat") is not None else None,
                       "objvel": p["obj"]["vel"][0].float().cpu().numpy() if "vel" in p["obj"] else None}
            if getattr(model, "spec_1m", None):                               # the 1 m centre-line head: its presence is a layer of `prob`, its direction goes to the angle check
                p1, dts[dt]["dir"] = A.predictions_1m(p, 0, model.spec_1m)
                prob.update(p1)
        return dts
''', ''),
    ('''The outputs are built by
the same functions as BEVSubject.predict builds them (`predictions`, `predictions_1m`, `object_head.decode`, `model.boxes`) and have the same structure and dtypes:
tests/test_bev_free.py checks them bit for bit against the live subject on real frames. The five lines of output building are copied from bev_adapter.BEVSubject.predict:
keep them in step with it (that test fails when they drift). One network per process''', '''The outputs are built by
the same function as BEVSubject.predict builds them (`bev_adapter.outputs_of`) and have the same structure and dtypes:
tests/test_bev_free.py checks them bit for bit against the live subject on real frames. One network per process'''),
])
print("ok")
