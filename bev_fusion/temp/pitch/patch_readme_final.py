def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


ARCH = '''### Architecture
```
 command line   python -m evalsuite <Class.method> ...        (suitecli.py: every method = one command, --json, exit 0 / 1 / 2)
      |
      v
 Evaluation (api.py)   scores and reads
   run  score  check  show_numbers  split_numbers  find_worst  compare  get_cost  report  make_card  draw_frame  list  list_sets
      |  calls the Runner (injectable: Evaluation(runner=...))                        |  scores through
      v                                                                                v
 Runner (runner_api.py)   the only class that executes the network          engine: Quantitative.run(results=...)
   run  check  tap  list  time  probe          no ground truth, no loss,      -> RunStore  <net>/eval/suite/quant/<e-id>
   always warm (D70)                                                            (additive counts per frame: small)
      |  writes / reads
      v
 ResultSet (results.py)   <net>/eval/results/<r-id>/   frames/<scene>_<i>.npz   taps/<scene>_<i>/<tap>.npy   manifest.json
   one per identity (network, checkpoint, inputs); any question reuses its frames; derived data

 project adapter  <project>/evaluation/evalsuite.json -> bev_eval.py
   required: name out network sets identity source subject task
   optional: free_source free_subject gt_source reference card time probe resolve release inputs MEMORY_TAP COMPARE_IGNORE
   BEV: bev_adapter (frozen), bev_free (the network without labels), bev_suite, bev_sets (the sets = the training's splits/)
```
Data flow of one `Evaluation.run`: frames (a text) -> `Frames.resolve` -> the Runner runs the scenes up to their last kept frame, whole and warm, and stores the kept frames as a ResultSet
(the temporary data: only what the network predicted) -> the Evaluation scores them against the ground truth (`reference`, no inputs) into a RunStore (the counts) -> `show_numbers` /
`compare` / `find_worst` read the counts. The ResultSet is only needed to score again (another ground truth, metric or subset) and to draw; it is derived data (deletable through
`autoresearch/tools/package.py`, D40).

### Command line
| what | command (from `autoresearch/`, `AR_PROJECT=../bev_fusion`) |
|---|---|
| the dry run | `python -m evalsuite check runs/t23_C_s1 small.val` (the Evaluation's; `Runner.check` is the network's alone) |
| run and score | `python -m evalsuite run runs/t23_C_s1 small.val --gt-key v8 [--metrics maps,boxes,fusion] [--taps fused] [--device cpu]` |
| predict only, no labels | `python -m evalsuite Runner.run runs/t23_C_s1 small.val --taps fused` |
| read | `show-numbers ID [--match M]`, `split-numbers ID [--axis scene]`, `find-worst ID`, `compare A B`, `get-cost ID`, `report ID`, `make-card ID`, `draw-frame ID FRAME` |
| what exists | `list-sets`, `Evaluation.list NETWORK`, `tap RESULTS FRAME [--name N]`, `methods` |
| every command | a bare word is the Evaluation's method when it has one, else the Runner's (`list`, `probe`, `selftest` and `Runner.run` need the class name); `--json` (nothing else on stdout), `--project`, `--adapter`; exit 0 ok, 1 `check` found problems, 2 any error (`EVALSUITE_TRACEBACK=1` shows the traceback) |
'''

patch("evalsuite/README.md", [
    ('''r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # predict what is not stored yet (warm, D70), score it: a Result
r.quant();  r.by("scene");  r.worst("seg", n=5);  r.cost();  r.report();  r.card()
ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids''',
     '''r = ev.run("runs/t23_C_s1", "small.val", gt_key="v8")               # predict what is not stored yet (warm, D70), score it: a dict with the id
ev.show_numbers(r["id"]);  ev.split_numbers(r["id"], "scene");  ev.find_worst(r["id"], "seg", n=5);  ev.get_cost(r["id"]);  ev.report(r["id"]);  ev.make_card(r["id"])
ev.compare(control_id, r["id"])                                     # b - a, paired bootstrap; one network per process: the control was run in another one, compare by ids'''),
    ("Only `predict` / `run` (and `time`, `probe`, `qual_frame`) execute the network;", "Only the Runner (`Runner.run`, `time`, `probe`; `Evaluation.run` calls it) and `draw_frame` execute the network;"),
    ("(only `qual_frame`, which draws a frame, runs the network again)", "(only `draw_frame`, which draws a frame, runs the network again)"),
    ("`Runner.run` then `score`: a Result (a dict with the id, whose methods read the evaluation) |", "`Runner.run` then `score`: a plain dict (the id, what was done, the warnings) |"),
    ("| `quant(id, match)`, `by(id, axis)`, `worst(id, match, n)` | `quant ID`, `by ID`, `worst ID` | the numbers, the numbers per scene or frame, where they come from |",
     "| `show_numbers(id, match)`, `split_numbers(id, axis)`, `find_worst(id, match, n)` | `show-numbers ID`, `split-numbers ID`, `find-worst ID` | the numbers, the numbers per scene or frame, where they come from |"),
    ("| `cost(id)`, `report(id)`, `card(id)` | `cost ID`, `report ID`, `card ID` | ms per frame, the html page, the metric card in the project's contract |",
     "| `get_cost(id)`, `report(id)`, `make_card(id)` | `get-cost ID`, `report ID`, `make-card ID` | ms per frame, the html page, the metric card in the project's contract |"),
    ("| `qual_frame(id, frame, view)` | `qual-frame ID FRAME` | one picture (the network runs on that frame with its scene's history) |", "| `draw_frame(id, frame, view)` | `draw-frame ID FRAME` | one picture (the network runs on that frame with its scene's history) |"),
    ("| `sets()`, `list(network)` | `sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |\n",
     "| `list_sets()`, `list(network)` | `list-sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |\n\n" + ARCH.rstrip("\n") + "\n"),
])
patch("../bev_fusion/evaluation/README.md", [
    ("python -m evalsuite Evaluation.quant t23_C_s1@e3f9a2c1d4b0 --match drivable  # the ids are printed by run and by `Evaluation.list runs/t23_C_s1`",
     "python -m evalsuite Evaluation.show-numbers t23_C_s1@e3f9a2c1d4b0 --match drivable   # the ids are printed by run and by `Evaluation.list runs/t23_C_s1`"),
    ("r.quant(), r.by(\"scene\"), r.worst(\"seg\"), r.card()", "ev.show_numbers(r[\"id\"]), ev.split_numbers(r[\"id\"], \"scene\"), ev.find_worst(r[\"id\"], \"seg\"), ev.make_card(r[\"id\"])"),
])
print("ok")
