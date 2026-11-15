# lanegraph/tests - `run_lg_tests.py` runs them all (exit 1 on a failure)

| file | what |
|---|---|
| `run_lg_tests.py` | the runner: `--quick` (unit tests only), `--seeds`, `--frames`, `--stride` (strided VAL rows), `--fuzz` |
| `test_lg_readout.py` | the hard output steps on toy frames: suppression, chains, Douglas-Peucker, snap, angle gate, cover, allowed, attached ends |
| `test_lg_equivalence.py` | exact equality with the original modules stage by stage and of the delivered graphs (incl. kinds, Douglas-Peucker vs `_dp_keep`), a read-out fuzz against the original read-out, golden digests |
