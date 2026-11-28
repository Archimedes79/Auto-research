"""pmap.py - an order-keeping parallel map over a fork pool, for CPU scoring (the scorer's cells and frame chunks are independent; measured 2.5x with 3
processes on a loaded host, numbers identical).

    from lane_center.evaluation.pmap import pmap
    rows = pmap(lambda cell: score(cell), cells, jobs=3)        # jobs <= 1: a plain loop in this process (the code path of before)

`fn` may be a closure over big data (the dump, the frames): the pool is forked after `fn` and the items are put into module globals, so nothing but the index
goes to a worker and only the result comes back (it must be picklable).
Order = the order of `items`; an exception in a worker is raised here; a call inside a worker runs serially (no nested pools). Linux (fork) only: elsewhere it
falls back to the serial loop. Test: `python -P -m lane_center.evaluation.pmap` (selftest).
"""
import multiprocessing as mp
import os

_FN = None
_ITEMS = None
_IN_WORKER = False


def _init():
    global _IN_WORKER
    _IN_WORKER = True


def _call(i):
    return _FN(_ITEMS[i])


def pmap(fn, items, jobs=1):
    """[fn(x) for x in items], in `jobs` forked processes when jobs > 1 and there is more than one item"""
    global _FN, _ITEMS
    items = list(items)
    jobs = min(int(jobs), len(items))
    if jobs <= 1 or _IN_WORKER or "fork" not in mp.get_all_start_methods():
        return [fn(x) for x in items]
    _FN, _ITEMS = fn, items
    try:
        with mp.get_context("fork").Pool(jobs, initializer=_init) as pool:
            return pool.map(_call, range(len(items)), chunksize=1)
    finally:
        _FN = _ITEMS = None


def selftest():
    import numpy as np
    big = np.arange(1_000_000, dtype=np.float64)
    f = lambda k: float(big[k * 1000:(k + 1) * 1000].sum()) + os.getpid() * 0           # a closure over big data; the pid term never changes the value
    ser = pmap(f, range(7), jobs=1)
    par = pmap(f, range(7), jobs=3)
    assert ser == par == [float(big[k * 1000:(k + 1) * 1000].sum()) for k in range(7)], "serial and pooled results must be identical, in order"
    pids = pmap(lambda _: os.getpid(), range(6), jobs=3)
    assert len(set(pids)) > 1 and os.getpid() not in pids, "jobs 3 must use other processes"
    assert pmap(lambda _: os.getpid(), range(3), jobs=1) == [os.getpid()] * 3, "jobs 1 stays in this process"
    assert pmap(lambda x: x, [], jobs=3) == [] and pmap(lambda x: x + 1, [4], jobs=3) == [5], "empty and single item"
    nested = pmap(lambda k: pmap(lambda x: os.getpid(), range(2), jobs=2), range(2), jobs=2)
    assert all(len(set(n)) == 1 for n in nested), "a call inside a worker runs serially"
    try:
        pmap(lambda x: 1 // x, [1, 0, 2], jobs=2)
        raise AssertionError("a worker's exception must be raised")
    except ZeroDivisionError:
        pass
    assert _FN is None and _ITEMS is None, "the globals are released"
    print("pmap selftest ok")


if __name__ == "__main__":
    selftest()
