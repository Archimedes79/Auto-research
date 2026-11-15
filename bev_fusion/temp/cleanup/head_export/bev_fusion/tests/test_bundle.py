"""The network is a closed bundle: training/model.py and training/losses.py import, transitively, no data loader and no trainer. CPU, one second, no
data. The rule is the provenance tool's own (autoresearch/tools/provenance.py `import_closure`: imports in a `_selftest` / `_cli` / `main`
function or an `if __name__ == "__main__"` block are listed, not followed): an import of train.py, nuscenes_dataset.py or feature_cache.py
anywhere else in the closure is a real edge of the network and fails here - the model bundle of a run (`provenance.py bundle`) would carry
or name it. Mutation: the closure of a copy of the folder in which precompute.py imports `train` inside a plain function must be refused.
Run (from the bev_fusion folder): wsl ~/miniconda3/envs/gcar/bin/python tests/test_bundle.py"""
import os
import shutil
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "..", "autoresearch", "tools"))
import provenance as P  # noqa: E402


def edges(code_dir):
    info = {}
    files = P.import_closure(code_dir, exclude=P.BUNDLE_EXCLUDE, info=info)
    return files, sorted(info["skipped"])


def main():
    code = os.path.join(ROOT, "training")
    files, skipped = edges(code)
    assert {"model.py", "losses.py", "precompute.py", "extract.py", "temporal_fusion.py"} <= set(files), files
    assert not skipped, f"the network imports a data loader or the trainer outside a test scope: {skipped}"
    tmp = tempfile.mkdtemp(prefix="bundle_mutant_", dir=os.path.join(ROOT, "temp") if os.path.isdir(os.path.join(ROOT, "temp")) else None)
    try:
        mutant = os.path.join(tmp, "training")
        shutil.copytree(code, mutant, ignore=shutil.ignore_patterns("__pycache__", "legacy", "configs"))
        with open(os.path.join(mutant, "precompute.py"), "a", encoding="utf-8") as f:
            f.write("\n\ndef network_inputs(config):\n    import train as T\n    return T.read_config(config)\n")
        _, bad = edges(mutant)
        assert any(e.endswith("-> train.py") and e.startswith("precompute.py") for e in bad), \
            f"a function-level import of train.py in precompute.py must be found: {bad}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"test_bundle passed: the closure of model.py + losses.py is {len(files)} files and imports no loader or trainer outside a test scope; "
          f"a function-level `import train` in precompute.py is caught")


if __name__ == "__main__":
    main()
