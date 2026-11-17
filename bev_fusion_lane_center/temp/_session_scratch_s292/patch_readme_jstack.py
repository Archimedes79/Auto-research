import io

p = "evaluation/README.md"
lines = io.open(p, encoding="utf-8").read().split("\n")
idx = [i for i, l in enumerate(lines) if l.startswith("| `jwire_verdict.py` |")]
assert len(idx) == 1
row = "| `jstack.py` | E363 (promoted from the scratch `temp/e363/make_run.py`, `faithful.py`, `surgery_check.py`): the pair run of E324 s read on ANOTHER graph's node head: `surgery <c2|a|b> <seed>` (run folder `runs/e363_<g>_s<S>`: the checkpoint of the pair run with the 38 `nodes.*` tensors of B' (b = B itself, the fixture) + `provenance.json`), `dump --run` (`jends_eval.dump` unchanged, with the Matrix-NMS decay of B' in front of `Attach` for arm a, which is NOT inside `nodes.*`), `check-nodes` (the pair model's node outputs against `QGraph.node_stage` of B', GPU), `check-arrays` (tensors, provenance sha256, dump arrays bit for bit, the fixture), `selftest` (CPU, toy tensors: swap, pyramid check, array comparison, layer wrapper against `qmnms.matrix_nms_logit`, layer parameters; 16 mutants of the pure parts killed). The scoring of such a run folder is `jends_eval.py score --run e363_<g>_s<S> --full / --base / --spec ...` (card E363). |"
lines.insert(idx[0] + 1, row)
io.open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
