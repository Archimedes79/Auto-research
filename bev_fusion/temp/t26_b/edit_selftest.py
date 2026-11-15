p = "evaluation/bev_selftest.py"
s = open(p, encoding="utf-8").read()
s = s.replace("            s.task.junction_free, s.task._junction = True, (lambda g: inter)\n", "            s.task.junction = lambda g: inter\n")
anchor = '''@check("bev_probe: the chain's names resolve lazily'''
new = '''class MaskedSource(SynthSource):
    """SynthSource that describes itself with the mask id of the junction-free rows, as the BEV adapter's Source does after bev_suite.make_suite"""
    def __init__(self, mask, **kw):
        super().__init__(**kw)
        self.mask = mask

    def describe(self):
        return dict(super().describe(), junction_mask=self.mask)


@check("--junction-free mask id (D66): the BEV Source describes itself with it only when set; the store's identity rule refuses a resume with another mask id and `compare` two stores with "
       "different ones; the same id resumes")
def _():
    from types import SimpleNamespace as NS
    from Evaluation.store import RunStore

    class Nusc:
        def get(self, table, tok):
            return {"scene_token": "s", "timestamp": int(tok[1:])} if table == "sample" else {"name": "scene-0001"}

    class Ds:
        samples, nusc, grid, gt_ss, gt = ["t0", "t1"], Nusc(), NS(res=0.5), 4, NS(signature=lambda: "sig")

        def __len__(self):
            return 2
    src = A._make_source_class()(NS(version="v1.0-mini", feature_cache=None), Ds(), NS(T=NS(collate=None, collate_cached=None)), workers=0, evaluator="abcdef012345")
    assert "junction_mask" not in src.describe(), "no option, no key: the stored descriptions of every other evaluation are unchanged"
    mask = {"lane_store": "s1", "raster": "1", "borders_m": {"fovj": 0.0, "fovjb": 0.5}, "cache_data": {"fovj": "d", "fovjb": "d"}}
    src.junction_mask = mask
    assert src.describe()["junction_mask"] == mask
    groups = ["vehicle", "two_wheeler", "pedestrian"]
    with tempfile.TemporaryDirectory() as d:
        def suite_of(m):
            return EvalSuite(MaskedSource(m), SynthSubject(), BEVTask(GRID, SPEC, groups, refine=True), out=d, state="reset")
        first = suite_of(mask).quant.run(scenes=["bev-00"], name="x", progress=0)
        assert first.manifest["source"]["junction_mask"] == mask and first.manifest["finished"]
        assert suite_of(dict(mask)).quant.run(scenes=["bev-00"], name="x", progress=0).manifest["resumed"] == 1, "the same mask id resumes"
        other = dict(mask, lane_store="s2")
        for other_mask in (other, dict(mask, cache_data={"fovj": "d2", "fovjb": "d"})):
            try:
                suite_of(other_mask).quant.run(scenes=["bev-00"], name="x", progress=0)
                raise AssertionError("a resume with another mask id must be refused")
            except ValueError as e:
                assert "junction_mask" in str(e), e
        suite_of(other).quant.run(scenes=["bev-00"], name="y", progress=0)
        try:
            suite_of(mask).quant.compare("x", "y", boot=20)
            raise AssertionError("two stores with different mask ids must not be compared")
        except ValueError as e:
            assert "junction_mask" in str(e), e
        RunStore.create(os.path.join(d, "quant", "x"), dict(first.manifest))                         # and the store's own rule says yes to its own manifest


'''
assert anchor in s
s = s.replace(anchor, new + anchor, 1)
open(p, "w", encoding="utf-8").write(s)
