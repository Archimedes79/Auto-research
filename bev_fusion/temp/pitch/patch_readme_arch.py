def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


ARCH = '''### Architecture
```
 command line   python -m evalsuite <Class.method> ...      (suitecli.py: every method = one command, --json, exit 0 / 1 / 2)
      |
      v
 Evaluation (api.py)  -- scores and reads --------------------------------------------.   reads
   run  score  check  quant  by  worst  compare  cost  report  card  qual_frame  list   |
      | calls (injectable: Evaluation(runner=...))                                      v
      v                                                                      engine: Quantitative.run(results=...)
 Runner (runner_api.py)  -- the only class that executes the network               -> RunStore  <net>/eval/suite/quant/<e-id>
   run  check  tap  list  time  probe        (no ground truth, no loss; always warm)      (additive counts per frame)
      | writes / reads
      v
 ResultSet (results.py)    <net>/eval/results/<r-id>/   frames/<scene>_<i>.npz   taps/<scene>_<i>/<tap>.npy   manifest.json
                           one per identity (network, checkpoint, inputs); any question reuses its frames

 project adapter (<project>/evaluation/evalsuite.json -> bev_eval.py):  name out network sets identity source subject task [free_source free_subject
   gt_source reference card time probe resolve release inputs MEMORY_TAP COMPARE_IGNORE]     BEV: bev_adapter, bev_free, bev_suite, bev_sets
```
Data flow of one `Evaluation.run`: frames (text) -> `Frames.resolve` -> the Runner runs the scenes up to their last kept frame, whole and warm, and stores the kept frames as a ResultSet
(the temporary data: only what the network predicted) -> the Evaluation scores them against the ground truth (`reference`, no inputs) into a RunStore (the counts: small) -> `quant` / `compare` /
`worst` read the counts. The ResultSet is only needed to score again (another ground truth, metric or subset) and to draw; it is derived data (deletable through `autoresearch/tools/package.py`, D40).

### Command line
| what | command (from `autoresearch/`, `AR_PROJECT=../bev_fusion`) |
|---|---|
| the dry run | `python -m evalsuite check runs/t23_C_s1 small.val` (the Evaluation's; `Runner.check` is the network's alone) |
| run and score | `python -m evalsuite run runs/t23_C_s1 small.val --gt-key v8 [--metrics maps,boxes,fusion] [--taps fused] [--device cpu]` |
| predict only, no labels | `python -m evalsuite Runner.run runs/t23_C_s1 small.val --taps fused` |
| read | `quant ID [--match M]`, `by ID [--axis scene]`, `worst ID`, `compare A B`, `cost ID`, `report ID`, `card ID`, `qual-frame ID FRAME` (a bare word is the Evaluation's, else the Runner's; `quant`, `list`, `probe` need `Evaluation.` / `Runner.`) |
| what exists | `sets`, `Evaluation.list NETWORK`, `tap RESULTS FRAME [--name N]`, `methods` |
| every command | `--json` (nothing else on stdout), `--project`, `--adapter`; exit 0 ok, 1 `check` found problems, 2 any error (`EVALSUITE_TRACEBACK=1` shows the traceback) |

'''

patch("evalsuite/README.md", [
    ("| `sets()`, `list(network)` | `sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |\n",
     "| `sets()`, `list(network)` | `sets`, `Evaluation.list NETWORK` | the sets a project defines, what is stored for a network (predictions and evaluations) |\n\n" + ARCH.rstrip("\n") + "\n"),
])
print("ok")
