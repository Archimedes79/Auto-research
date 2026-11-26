"""The prediction step of a lane run: the live dump of the run's network on its split (jtoken_eval.dump --live: B' at batch 4, the GPU parity
of its token cache; the head at batch 8), written as <run>/token_<split>_live.pkl. The evaluation step (evaluation/adapter.py) decodes and scores
the dump; nothing here computes a number.

    python evaluation/lane_predict.py <the run's config.json>          (the worker starts it; RESEARCH_SPLIT names the split, default val)

A run that trains nothing (a measurement of an existing network) names it in the config: import_weights = "runs/<folder>" (may hold {seed});
its model.pt is copied byte for byte into the run's folder first, since its checks (B''s sha256, the token cache) read the file. A stored dump is
never overwritten: the step is done when it exists."""
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for sub_ in ("evaluation", "training", "net/prep", "net/model", "net/loss", "training/data", "training/data/prep", "training/train", "../.."):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import bflc_paths as BP  # noqa: E402
import jtoken_eval as JE  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def import_weights(cfg, run_dir):
    """the model.pt of `import_weights` into the run's folder, the same bytes"""
    src = os.path.join(BP.resolve(cfg["import_weights"].format(seed=cfg["seed"])), "model.pt")
    dst = os.path.join(run_dir, "model.pt")
    if not os.path.isfile(dst):
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
    if sha256(src) != sha256(dst):
        raise SystemExit(f"{dst} is not the model.pt of {src}: move it away")
    return src


def main(argv):
    cfg = json.load(open(argv[0], encoding="utf-8"))
    runs = os.path.normpath(cfg["out_root"])
    if runs != os.path.normpath(BP.RUNS):
        raise SystemExit(f"the run's out_root {runs} is not the problem's run root {BP.RUNS}")
    run_dir = os.path.join(runs, cfg["name"])
    os.makedirs(run_dir, exist_ok=True)
    if cfg.get("import_weights"):
        print(f"weights of {import_weights(cfg, run_dir)}", flush=True)
    a = argparse.Namespace(run=cfg["name"], seed=cfg["seed"], split=os.environ.get("RESEARCH_SPLIT") or "val", live=True, bs=None, every=1, drop=0.0,
                           drop_seed=0, max=0, cpu=False)
    path = JE.dump_path(a, BP.RUNS, 0)
    if os.path.exists(path):
        print(f"stored: {path}")
        return 0
    JE.dump(a, runs=BP.RUNS)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
