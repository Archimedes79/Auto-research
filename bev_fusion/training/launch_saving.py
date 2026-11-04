"""Switches of the launch- and sync-saving variants of the step (D53, Mirko 2026-11-04: "Yes, implement 1-4" of records/speed_options_2026-11-03.md).

The step is host-bound (about 45 thousand operators and a few hundred host syncs per step; the GPU kernels are busy a third of the time), so the variants below
do the same arithmetic with fewer launches or fewer waits for the GPU. Every switch is OFF by default: a stored config trains bit for bit as before. A variant is
equal to its plain twin bit for bit on the CPU in float32 / float64 (tests/test_launch_saving.py; in bf16 / fp16 the CPU half-precision scatter rounds differently for one call of 4N entries than for
four calls, about one ulp: no stored config uses half precision); on the GPU the scatter sums of `splat` add in an order the hardware chooses, which they do in the plain
twin too, so there it is equal up to rounding.

    splat   ModelConfig.splat_fused   one scatter over the four bilinear corners instead of four (splat.py)
    sync    ModelConfig.sync_free / LossConfig.sync_free   no data-dependent index (boolean mask, nonzero, int() of a tensor) where the same numbers come out of a
            fixed-shape or host-side form: the radar raster (motion_block.py), the memory re-centring (temporal_fusion.py), the motion term (losses.py), the
            distance-target discs (layers.py)
    (ModelConfig.share_mover is not here: it is a model-level choice, one splat for both heads, model.py / timed_head.py)

The switches are per-thread state set for the duration of one forward (`BEVFusionModel.forward`) or one loss call (`BEVLoss.forward`) with `enabled(...)`, so
the arithmetic of a call never depends on the order of earlier calls; outside such a block everything is off.
Run the self-test: python training/launch_saving.py
"""
import contextlib
import threading

NAMES = ("splat", "sync")


class _State(threading.local):                         # per thread: two threads that run a forward at the same time cannot switch each other's variants on or off
    def __init__(self):
        for k in NAMES:
            setattr(self, k, False)


_STATE = _State()


def on(name):
    if name not in NAMES:
        raise ValueError(f"unknown launch-saving switch {name!r}; known: {list(NAMES)}")
    return getattr(_STATE, name)


@contextlib.contextmanager
def enabled(**flags):
    """inside the block the named switches have these values (others keep theirs); restored on exit, also after an error"""
    unknown = set(flags) - set(NAMES)
    if unknown:
        raise ValueError(f"unknown launch-saving switch(es) {sorted(unknown)}; known: {list(NAMES)}")
    old = {k: getattr(_STATE, k) for k in NAMES}
    for k, v in flags.items():
        setattr(_STATE, k, bool(v))
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(_STATE, k, v)


def _selftest():
    assert not on("splat") and not on("sync")
    with enabled(splat=True):
        assert on("splat") and not on("sync")
        with enabled(sync=True):
            assert on("splat") and on("sync")
        assert on("splat") and not on("sync")
    assert not on("splat") and not on("sync")
    try:
        with enabled(splat=True, nope=True):
            raise AssertionError("an unknown switch was accepted")
    except ValueError:
        pass
    assert not on("splat"), "an error inside the block restores the state"
    try:
        with enabled(sync=True):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert not on("sync")
    print("launch_saving selftest ok: nested blocks restore the state, an unknown name is refused, an error restores it")


if __name__ == "__main__":
    _selftest()
