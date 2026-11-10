"""Start a training from a config file (the BEV system's `train.py <config.json>`, adapted): the one command that `python -m trainsuite start <config> --id <id>` wraps in `run.py`.

    wsl bash tools/env/_py14.sh run_config.py <config.json>        (the keys of the design: `training/startable.py`, `training/configs/README.md`)
    wsl bash tools/env/_py14.sh run_config.py --selftest

The config names its design; this file turns it into the argument line of the trainer of that design (`grid_head` -> `grid_train.py`, `qgraph_head` -> `qgraph_train.py`), so a trainer keeps one command line and a run is
one json file. The same command again resumes the run (the trainer's checkpoint).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
for sub in ("training", "training/train"):
    sys.path.insert(0, os.path.join(ROOT, sub))

FLAGS = {"grid_head": ("name", "seed", "steps", "bs", "lr", "width", "workers", "budget", "per100"),
         "qgraph_head": ("name", "seed", "phases", "steps", "bs", "workers", "width", "head", "resume_from", "gate_phases", "ev_w", "node_w_evt", "node_w_inter", "long_match", "tier_w", "vis_w",
                         "neg_w", "link_tier_w", "sparse", "node_loss", "zone_bias", "soft_nms")}
SWITCHES = ("smoke", "no_gates")                    # a key that is true becomes a flag without a value


def argv_of(cfg):
    """the argument list of the trainer of cfg["design"] (a list of strings); a key `node_w_evt` is the flag `--node-w-evt`"""
    design = cfg.get("design")
    if design not in FLAGS:
        raise SystemExit(f"design {design!r} is not one of {sorted(FLAGS)}")
    out = []
    for k in FLAGS[design]:
        if k not in cfg:
            continue
        v = cfg[k]
        out += ["--" + k.replace("_", "-")] + ([str(x) for x in v] if isinstance(v, list) else [str(v)])
    for k in SWITCHES:
        if cfg.get(k):
            out.append("--" + k.replace("_", "-"))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        a = argv_of(dict(design="grid_head", name="e271_a_s1", seed=1, steps=4000, bs=8, lr=1e-3, width=48, workers=3, budget=[11.2, 8.0, 14.3], per100=11.2, smoke=True))
        assert a == ["--name", "e271_a_s1", "--seed", "1", "--steps", "4000", "--bs", "8", "--lr", "0.001", "--width", "48", "--workers", "3", "--budget", "11.2", "8.0", "14.3", "--per100", "11.2",
                     "--smoke"], a
        assert "--smoke" not in argv_of(dict(design="grid_head", name="x", steps=3))
        q = argv_of(dict(design="qgraph_head", name="e273_p_s1", seed=1, phases=[3, 4], steps=[10, 20], node_w_evt=3.0, tier_w=[1, 2, 3], no_gates=True, workers=2))
        assert q == ["--name", "e273_p_s1", "--seed", "1", "--phases", "3", "4", "--steps", "10", "20", "--workers", "2", "--node-w-evt", "3.0", "--tier-w", "1", "2", "3", "--no-gates"], q
        try:
            argv_of(dict(design="nope"))
            raise AssertionError("an unknown design must stop")
        except SystemExit:
            pass
        print("run_config selftest passed")
        return
    import startable as ST
    cfg = json.load(open(ST.resolve(sys.argv[1]), encoding="utf-8"))
    problems, notes = ST.check(cfg)
    for n in notes:
        print("note:", n, flush=True)
    if problems:
        raise SystemExit("not startable: " + "; ".join(problems))
    mod = {"grid_head": "grid_train", "qgraph_head": "qgraph_train"}[cfg["design"]]
    trainer = __import__(mod)
    sys.argv = [mod + ".py"] + argv_of(cfg)
    trainer.main()


if __name__ == "__main__":
    main()
