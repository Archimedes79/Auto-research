old = open("evalsuite/api.py", encoding="utf-8").read()
marker = "# ------------------------------------------------------------------------------------------------ the class\n"
head = old[:old.index(marker)]
cls = open("../bev_fusion/temp/pitch/api_class2.py", encoding="utf-8").read()


def rep(old_, new_):
    global head
    assert head.count(old_) == 1, (head.count(old_), old_[:80])
    head = head.replace(old_, new_)


rep('''`run` is `predict` + `score`. Only `predict` (hence `run`) and `time`, `probe`, `qual_frame` execute the network; everything else reads what was stored. D70: a network with a memory always
runs warm: whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored, a frame already stored is not predicted again (a subset
of an earlier selection costs no network).''', '''`run` is `predict` + `score`. The class that executes a network is the Runner (runner_api.py, D72: it reads no ground truth); the Evaluation calls it (`Evaluation(runner=...)` injects another),
`predict` is its `run`. Everything else here reads what was stored (only `qual_frame`, which draws a frame, runs the network on it again). D70: a network with a memory always
runs warm: whole scenes in time order from the network's own start; the frames you ask for decide which results are kept and scored, a frame already stored is not predicted again (a subset
of an earlier selection costs no network).''')
rep('''  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device), MEMORY_TAP = the name of a tap whose mean is the fill of the
  network's memory (the health check: below).
Intermediate results: `predict(..., taps=("fused", ...))` stores the named taps of the Subject (its `taps()`) with the predictions; a frame stored without a tap is predicted again when the
tap is asked for. The health of the memory (D70) is checked on every scene the network runs, from the MEMORY_TAP means: empty at the first frame, growing over the frames; a scene that
fails is a warning in the result, never an error.''', '''  optional: resolve(text) -> network path (a run name, a relative path), release(source), reference(source, ref) -> the ground truth of a frame without loading the inputs,
  inputs(source) -> what the source was fed, COMPARE_IGNORE, card(...), time(network, n), probe(network, frames, device); free_source(network, of, workers, control) and
  free_subject(network, source, device): a Source without ground truth and a Subject that only predicts, which the Runner uses (else it uses source / subject); gt_source(...): the Source
  the Evaluation scores with (else source); MEMORY_TAP = the name of a tap whose mean is the fill of the network's memory (the health check, runner_api.py).
Intermediate results: `run(..., taps=("fused", ...))` stores the named taps of the Subject (its `taps()`) with the predictions; a frame stored without a tap is predicted again when the
tap is asked for.''')
rep("import importlib\nimport json\nimport os\nimport sys\nimport time\n", "import importlib\nimport json\nimport os\nimport sys\nimport time\n")
rep("import numpy as np\n\nfrom .contracts import FrameRef, Prediction, Source\n", "from .contracts import FrameRef, Source\n")
rep("from .results import Locked, ResultSet, digest, result_id, writer_lock\nfrom .runner import Runner\nfrom .sampling import Sampler\n", "from .results import Locked, ResultSet, digest, writer_lock\nfrom .sampling import Sampler\n")
open("evalsuite/api.py", "w", encoding="utf-8").write(head + cls)
print("ok", len((head + cls).splitlines()))
