"""default slim_at_end False (Mirko 2026-11-01: keep the full data to be able to resume later): the same exact-string replacements that were
made in the working tree, applied to the HEAD blobs written by build_tree2.sh (head_<name>) -> new_<name>. Every old string must match once."""
import sys

D = sys.argv[1]


def once(t, old, new):
    assert t.count(old) == 1, (t.count(old), old[:80])
    return t.replace(old, new)


R = {
    "train.py": [
        ('''    slim_at_end: bool = True                     # a run that finishes keeps ONE checkpoint without optimiser state and RNG (about half the
                                                 # size) and no older ones: it evaluates and warm-starts, it cannot resume training.
                                                 # False = `keep` full resume checkpoints. Unfinished or stopped runs always keep full ones
''', '''    slim_at_end: bool = False                    # True: a run that finishes keeps ONE checkpoint without optimiser state and RNG (about half
                                                 # the size, no older ones): it evaluates and warm-starts but cannot resume training.
                                                 # False (default; Mirko 2026-11-01: keep the full data to resume later): `keep` complete ones
'''),
        ('''                          slim_at_end=False, device="cpu", model={"trunk_width": 0.25})       # (slimming has its own test, 2b)''',
         '''                          device="cpu", model={"trunk_width": 0.25})'''),
        ('''        assert ck["step"] == 4 and not any(k.startswith("model.camera.") for k in ck["state"])
        changed = any(''', '''        assert ck["step"] == 4 and not any(k.startswith("model.camera.") for k in ck["state"])
        assert "opt" in ck and "rng" in ck and not ck.get("slim"), "the default keeps complete checkpoints (Adam moments and RNG)"
        changed = any('''),
    ],
    "scratch.py": [
        ("The finished checkpoints of a training run are slimmed by train.py itself (TrainConfig.slim_at_end), see training/README.md.",
         "A finished training run keeps complete checkpoints (resume later); train.py slims them only on request (TrainConfig.slim_at_end), see training/README.md."),
    ],
    "training_README.md": [
        ("The checkpoints of a FINISHED run are slimmed by `train.py` itself (`TrainConfig.slim_at_end`, default on: one checkpoint without optimiser state and RNG, about half the size; a slim checkpoint evaluates and warm-starts but cannot resume training) | `python training/scratch.py --selftest` (11 checks on real files)",
         "A FINISHED run keeps complete checkpoints (`TrainConfig.keep`) so that it can be resumed later; `train.py` slims them only on request (`TrainConfig.slim_at_end`, default off since 2026-11-01: one checkpoint without optimiser state and RNG, about half the size; it evaluates and warm-starts but cannot resume training) | `python training/scratch.py --selftest` (16 checks on real files)"),
    ],
    "temp_README.md": [
        ("a run that finishes keeps one slim checkpoint (`slim_at_end`), never a pile of resume checkpoints.",
         "a run that finishes keeps its `keep` complete checkpoints so that it can be resumed later (`slim_at_end` is an opt-in that keeps one slim checkpoint)."),
    ],
}
for name, reps in R.items():
    t = open(f"{D}/head_{name}", newline="").read()
    for old, new in reps:
        t = once(t, old, new)
    open(f"{D}/new_{name}", "w", newline="").write(t)
    print("ok", name)
