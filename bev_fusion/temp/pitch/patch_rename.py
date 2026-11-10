import re


def sub(path, pairs, regex=()):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert old in s, (path, old[:90])
        s = s.replace(old, new)
    for pat, new in regex:
        s, n = re.subn(pat, new, s)
        assert n, (path, pat)
    open(path, "w", encoding="utf-8").write(s)


# ------------------------------------------------------------------------------------------------ api.py: verb-first names, no Result proxy class
s = open("evalsuite/api.py", encoding="utf-8").read()
cut = s.index("class Result(dict):")
s = s[:cut].rstrip() + "\n"
open("evalsuite/api.py", "w", encoding="utf-8").write(s)
sub("evalsuite/api.py", [
    ('''    r = ev.run("runs/t23_C_s1", "large.val", gt_key="v8")         # predict (the network runs once on what is missing), then score: a Result
    r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
    ev.compare(a_id, b_id)                                         # paired bootstrap, b - a (a = the control);   ev.qual_frame(r["id"], "scene-0061/12")''',
     '''    r = ev.run("runs/t23_C_s1", "large.val", gt_key="v8")         # predict (the network runs once on what is missing), then score: a dict with the id and what was done
    ev.show_numbers(r["id"]);  ev.split_numbers(r["id"], "scene");  ev.find_worst(r["id"], "seg", n=5);  ev.get_cost(r["id"]);  ev.report(r["id"]);  ev.make_card(r["id"])
    ev.compare(a_id, b_id)                                         # paired bootstrap, b - a (a = the control);   ev.draw_frame(r["id"], "scene-0061/12")'''),
    ("(only `qual_frame`, which draws a frame, runs the network on it again)", "(only `draw_frame`, which draws a frame, runs the network on it again)"),
    ("    def sets(self) -> Dict[str, Any]:", "    def list_sets(self) -> Dict[str, Any]:"),
    ("    def quant(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:", "    def show_numbers(self, id: str, match: Optional[str] = None) -> Dict[str, Any]:"),
    ("    def by(self, id: str, axis: str = \"scene\") -> Dict[str, Any]:", "    def split_numbers(self, id: str, axis: str = \"scene\") -> Dict[str, Any]:"),
    ("    def worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:", "    def find_worst(self, id: str, match: Optional[str] = None, n: int = 10) -> Dict[str, List[dict]]:"),
    ("    def cost(self, id: str) -> Dict[str, Any]:", "    def get_cost(self, id: str) -> Dict[str, Any]:"),
    ("    def card(self, id: str) -> Dict[str, Any]:", "    def make_card(self, id: str) -> Dict[str, Any]:"),
    ("    def qual_frame(self, id: str, frame: str, view: str = \"overview\", device: str = \"cpu\", workers: int = 0) -> Dict[str, Any]:", "    def draw_frame(self, id: str, frame: str, view: str = \"overview\", device: str = \"cpu\", workers: int = 0) -> Dict[str, Any]:"),
    ('''metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0, progress: int = 50) -> "Result":''', '''metrics: Sequence[str] = ("maps", "boxes"), workers: int = 0, progress: int = 50) -> Dict[str, Any]:'''),
    ('''            workers: int = 2, control: str = "", progress: int = 50) -> "Result":''', '''            workers: int = 2, control: str = "", progress: int = 50) -> Dict[str, Any]:'''),
    ("metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a Result: a dict with the id, whose methods read the evaluation", "metrics = what the project offers (BEV: maps, boxes, curves, fusion, velocity). -> a dict: the id, what was done, the warnings"),
    ("        return Result(self, {**info, \"warnings\"", "        return ({**info, \"warnings\""),
])
s = open("evalsuite/api.py", encoding="utf-8").read()
s = s.replace('''"seconds_network": seconds_network, "seconds_score": round(time.time() - t1, 1)})''', '''"seconds_network": seconds_network, "seconds_score": round(time.time() - t1, 1)})''')
open("evalsuite/api.py", "w", encoding="utf-8").write(s)

# ------------------------------------------------------------------------------------------------ the command line docstring and legacy words
sub("evalsuite/apicli.py", [
    ("`Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable`", "`Evaluation.show-numbers t23_C_s1@e3f9a2c1d4b0 --match drivable`"),
    ("`qual-frame` is `qual_frame`.", "`draw-frame` is `draw_frame`."),
    ('''LEGACY = ("list", "quant", "qual", "probe", "selftest", "selftest-adapter")''', '''LEGACY = ("list", "probe", "selftest", "selftest-adapter")'''),
    ("`list`, `quant`, `probe`, `selftest` need `Evaluation.`", "`list`, `probe`, `selftest` need the class name"),
])

# ------------------------------------------------------------------------------------------------ the toy selftest
sub("evalsuite/selftest_api.py", [
    ('r.quant()["numbers"], _engine(net, fr), spec', 'ev.show_numbers(r["id"])["numbers"], _engine(net, fr), spec'),
    ('one.by("frame")["demo-1/004"]["iou.iou"] == allr.by("frame")["demo-1/004"]["iou.iou"]', 'ev.split_numbers(one["id"], "frame")["demo-1/004"]["iou.iou"] == ev.split_numbers(allr["id"], "frame")["demo-1/004"]["iou.iou"]'),
    ('r2.quant() == allr.quant()', 'ev.show_numbers(r2["id"]) == ev.show_numbers(allr["id"])'),
    ('r3.quant()["numbers"]["iou.iou"] != allr.quant()["numbers"]["iou.iou"]', 'ev.show_numbers(r3["id"])["numbers"]["iou.iou"] != ev.show_numbers(allr["id"])["numbers"]["iou.iou"]'),
    ('"loss.total" not in r3.quant()["numbers"]', '"loss.total" not in ev.show_numbers(r3["id"])["numbers"]'),
    ('    c = low.compare(allr)\n', '    c = ev.compare(allr["id"], low["id"])\n'),
    ('low.quant()["numbers"]["iou.iou"] - allr.quant()["numbers"]["iou.iou"]', 'ev.show_numbers(low["id"])["numbers"]["iou.iou"] - ev.show_numbers(allr["id"])["numbers"]["iou.iou"]'),
    ('done.quant()["numbers"]', 'ev.show_numbers(done["id"])["numbers"]'),
    ('q = allr.quant("iou")', 'q = ev.show_numbers(allr["id"], "iou")'),
    ('set(allr.by("scene")) == {"demo-0", "demo-1", "demo-2"} and len(allr.by("frame")) == 18', 'set(ev.split_numbers(allr["id"], "scene")) == {"demo-0", "demo-1", "demo-2"} and len(ev.split_numbers(allr["id"], "frame")) == 18'),
    ('_raises(EvaluationError, "axis", allr.by, "range")', '_raises(EvaluationError, "axis", ev.split_numbers, allr["id"], "range")'),
    ('set(allr.worst(n=2))', 'set(ev.find_worst(allr["id"], n=2))'),
    ('allr.cost()["predict"]["n"] == 18 and allr.cost()["devices"] == ["cuda"]', 'ev.get_cost(allr["id"])["predict"]["n"] == 18 and ev.get_cost(allr["id"])["devices"] == ["cuda"]'),
    ('os.path.isfile(allr.report()) and allr.card()["name"] == allr["id"]', 'os.path.isfile(ev.report(allr["id"])) and ev.make_card(allr["id"])["name"] == allr["id"]'),
    ('ev.sets() == {"val": 3, "small": 1}', 'ev.list_sets() == {"val": 3, "small": 1}'),
    ('Evaluation(adapter=ME).quant(allr["id"])["numbers"] == allr.quant()["numbers"]', 'Evaluation(adapter=ME).show_numbers(allr["id"])["numbers"] == ev.show_numbers(allr["id"])["numbers"]'),
    ('pic = ev.qual_frame(allr["id"], "demo-1/4")', 'pic = ev.draw_frame(allr["id"], "demo-1/4")'),
    ('Evaluation(adapter=ME).quant(info["id"])["numbers"]["iou.iou"]', 'Evaluation(adapter=ME).show_numbers(info["id"])["numbers"]["iou.iou"]'),
    ('call("Evaluation.quant", info["id"], "--match", "iou", "--json", *base)', 'call("Evaluation.show_numbers", info["id"], "--match", "iou", "--json", *base)'),
    ('call("qual-frame", info["id"], "demo-0/2", *base)                         # the hyphen form of qual_frame', 'call("draw-frame", info["id"], "demo-0/2", *base)                         # the hyphen form of draw_frame'),
    ('call("Evaluation.quant", "thr0.5@e0000", *base)', 'call("Evaluation.show_numbers", "thr0.5@e0000", *base)'),
    ('_raises(EvaluationError, "not an evaluation id", ev.quant, allr["results"])', '_raises(EvaluationError, "not an evaluation id", ev.show_numbers, allr["results"])'),
    ('_raises(EvaluationError, "no evaluation", ev.quant, "thr0.5@edeadbeef0000")', '_raises(EvaluationError, "no evaluation", ev.show_numbers, "thr0.5@edeadbeef0000")'),
    ("    # the result is a plain dict when copied or pickled\n    assert type(copy.deepcopy(allr)) is dict and pickle.loads(pickle.dumps(allr)) == dict(allr)\n", "    assert type(copy.deepcopy(allr)) is dict and pickle.loads(pickle.dumps(allr)) == allr, \"the result of run is plain data\"\n"),
])

# ------------------------------------------------------------------------------------------------ the BEV gate
sub("tests/test_evaluation_api.py" if False else "../bev_fusion/tests/test_evaluation_api.py", [
    ('"Evaluation.card(" in out.stdout', '"Evaluation.make_card(" in out.stdout'),
    ('mine = ev.by(r["id"], "frame")', 'mine = ev.split_numbers(r["id"], "frame")'),
    ('q2, q3 = r.quant()["numbers"], r3.quant()["numbers"]', 'q2, q3 = ev.show_numbers(r["id"])["numbers"], ev.show_numbers(r3["id"])["numbers"]'),
    ('        c = r.card()', '        c = ev.make_card(r["id"])'),
])
print("ok")
