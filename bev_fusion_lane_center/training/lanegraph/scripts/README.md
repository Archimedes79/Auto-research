# lanegraph/scripts - command-line entry points (run from the problem folder through `tools/env/_py14.sh`)

| file | what |
|---|---|
| `lg_infer.py` | checkpoint + the first N VAL frames -> delivered graphs as JSON (Douglas-Peucker 0.3 m); `--attached-ends` for the second read-out |
| `lg_eval.py` | both read-outs from one forward, scored by `evaluation/qgraph_eval.py`, next to the stored per-frame counts of the cards; link-candidate counts per batch |
| `lg_timing.py` | ms per frame (batch 1) of the champion next to the original modules, alternating blocks; the read-out timed separately |
| `lg_project.py` | the project's code for the scripts and tests: path set-up, the original models, the original read-out of one frame |
