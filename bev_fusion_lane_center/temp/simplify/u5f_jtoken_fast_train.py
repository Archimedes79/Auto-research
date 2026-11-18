"""Trainer of the E365 token head on the speed path: `jtoken_train.py` with `jtoken_fast.FastJTokenHead` / `fast_loss` and the GPU-resident loader `lane_ptok_gpu`. Same arguments, same recipe, same run directory,
same `model.pt` / `ckpt.pt` and recipe (a run of this script resumes a plain run's checkpoint with the same arguments, and the other way round), same data order; start it through `autoresearch/tools/run.py` like `jtoken_train.py`.

    wsl bash tools/env/_pyr.sh jtoken_fast_train.py --name e365_x_s0 --seed 0 --ident e292_a --steps 12000 --bs 8 --lr 1e-3 --workers 4 --none-weight 0.25 [every other jtoken_train option] [--cpu-loader] [--compile-pairs]

`--cache <root>` may be relative to the problem folder (`lane_ptok_data.cache_dir` resolves it; a second K lives in its own root: data/build_tokcache_root.py).
Own flags (removed before `jtoken_train` parses the line): `--cpu-loader` keeps the CPU DataLoader of `lane_ptok_data` (the dense batches; the loss still runs on the fast path), `--compile-pairs` wraps the pair-bias
stack in torch.compile (the first step compiles, about a minute; the numbers agree with the eager stack to bf16 rounding). `--workers N` selects the data ORDER of a DataLoader with (N > 0, persistent) or without
workers, exactly as before, but no worker runs with the GPU loader. Refused with the GPU loader: `--distill-from` (it reads keys the GPU batches do not carry; use `--cpu-loader`).
The script patches the names `jtoken_train` resolves at call time (`build_head`, `JK.jtoken_loss`, `lane_ptok_data.loader`) in this process only; nothing in `jtoken_train.py` / `jtoken.py` is edited. The run folder gets
`fast_path.json` (which parts ran). The speed path is the same function: parity tests `grid/test_jtoken_fast.py`, `grid/test_lane_ptok_gpu.py`; the paired loss-trajectory check against the plain trainer is `grid/test_jtoken_fast.py`'s
and the card's (a speed change reproduces the reference numbers)."""
import json
import os
import sys

ROOT = os.environ.get("BFLC_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for sub_ in ("training", "training/grid", "training/data", "training/train", "training/data/prep", "evaluation"):
    sys.path.insert(0, os.path.normpath(os.path.join(ROOT, sub_)))
import jtoken as JK  # noqa: E402
import jtoken_fast as JF  # noqa: E402
import jtoken_train as JT  # noqa: E402
import trainer as T  # noqa: E402

OWN = ("--cpu-loader", "--compile-pairs")


def fast_build_head(args):
    """`jtoken_train.build_head` with the fast head class (the same arguments and weights)"""
    d, heads, layers, ffn = JT.arch(args)
    head = JF.FastJTokenHead(d=d, heads=heads, layers=layers, ffn=ffn, live_inputs=bool(args.get("live_inputs", JT.ARCH_DEFAULTS["live_inputs"])))
    return head.compile_pairs() if _OPT["compile"] else head


_OPT = dict(compile=False, cpu_loader=False)


def install(cpu_loader=False, compile_pairs=False):
    """patch the names `jtoken_train` resolves at call time (this process only)"""
    _OPT.update(compile=compile_pairs, cpu_loader=cpu_loader)
    JT.build_head = fast_build_head
    JK.jtoken_loss = JF.fast_loss
    load = JT.load_teachers

    def fast_teachers(args, dev):                                            # frozen teachers (E379) read no handle scales: skip the dense scale MLP in their forward
        heads = load(args, dev)
        for h in heads:
            h.scale_in_eval = False
        return heads

    JT.load_teachers = fast_teachers
    if not cpu_loader:
        import lane_ptok_data as LP
        import lane_ptok_gpu as LG
        LP.loader = LG.loader


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    own = {k: k in argv for k in OWN}
    argv = [x for x in argv if x not in OWN]
    a = JT.parse_args(argv)
    if not own["--cpu-loader"] and not a.smoke:
        assert not a.distill_from, "the GPU loader does not carry what --distill-from reads: add --cpu-loader"
    assert not getattr(a, "live_inputs", False), "the speed path is the plain head: live_inputs runs the parent's forward, which has no scale_at for the fast loss"
    import torch
    assert a.smoke or torch.cuda.is_available(), "no GPU visible: a run of the card trains on the GPU (bf16); only --smoke may run on the CPU"
    install(cpu_loader=own["--cpu-loader"] or a.smoke, compile_pairs=own["--compile-pairs"])
    note = dict(head="jtoken_fast.FastJTokenHead", loss="jtoken_fast.fast_loss", loader="lane_ptok_data (CPU)" if (own["--cpu-loader"] or a.smoke) else "lane_ptok_gpu (GPU-resident)", compile_pairs=own["--compile-pairs"])
    check = JT.check_start

    def check_then_note(args, folder):                                      # the run folder is written only after the recipe check accepted it
        check(args, folder)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "fast_path.json"), "w", encoding="utf-8") as fh:
            json.dump(note, fh)

    JT.check_start = check_then_note
    try:
        JT.train(a, T.RUNS)
    finally:
        JT.check_start = check


if __name__ == "__main__":
    main()
