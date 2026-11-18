"""Tests of evaluation/gpu_gate.py (D100): the nvidia-smi reading and its failures, the boundary of the limit, the count of waiting lines, the timeout and environment overrides, the status file,
and that `jends_eval.dump` and `jmaps_eval.score` call the gate before they touch the model. CPU only, no GPU, no CUDA call (nvidia-smi is replaced by a fake).
Run (from the problem folder): wsl bash tools/env/_py14.sh test_gpu_gate.py"""
import os
import subprocess
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.dirname(HERE)
sys.path.insert(0, EVAL)
import gpu_gate as GG  # noqa: E402


def _fake_smi(text=None, exc=None):
    def run(cmd, **kw):
        assert cmd[0] == "nvidia-smi" and "--query-gpu=memory.used" in cmd and kw.get("timeout")                      # no CUDA call, a timeout is set
        if exc:
            raise exc
        return types.SimpleNamespace(stdout=text)
    return run


def test_reading():
    old = subprocess.run
    try:
        subprocess.run = _fake_smi("11425\n")
        assert GG.used_mb() == 11425
        subprocess.run = _fake_smi("9000\n12000\n")                                                                  # several GPUs: the largest
        assert GG.used_mb() == 12000
        subprocess.run = _fake_smi("", None)                                                                         # nothing printed
        assert GG.used_mb() is None
        subprocess.run = _fake_smi(None, FileNotFoundError())                                                        # no nvidia-smi
        assert GG.used_mb() is None
        subprocess.run = _fake_smi(None, subprocess.TimeoutExpired("nvidia-smi", 20))
        assert GG.used_mb() is None
    finally:
        subprocess.run = old


def _clock():
    now = [0.0]
    return now, (lambda s: now.__setitem__(0, now[0] + s)), (lambda: now[0])


def test_boundary_and_waiting_lines():
    os.environ.pop("GPU_GATE_MB", None)
    os.environ.pop("GPU_GATE_TIMEOUT_S", None)
    now, sleep, clock = _clock()
    out = []
    lg = lambda s, **k: out.append(s)  # noqa: E731
    assert GG.wait_for_room(_used=lambda: 8999.0, _sleep=sleep, _now=clock, log=lg, status=None) == 0.0 and not out
    w = GG.wait_for_room(_used=lambda: 9000.0, timeout_s=0, _sleep=sleep, _now=clock, log=lg, status=None)          # the limit itself is not room
    assert w == 0.0 and any("going on" in s for s in out)
    now[0], out[:] = 0.0, []
    GG.wait_for_room(_used=lambda: 11600.0, poll_s=60, timeout_s=25 * 60, _sleep=sleep, _now=clock, log=lg, status=None)    # 25 polls: a line at poll 0, 10, 20 and the end
    assert sum("waiting" in s for s in out) == 3 and sum("going on" in s for s in out) == 1, out


def test_overrides_and_status_file():
    now, sleep, clock = _clock()
    out = []
    lg = lambda s, **k: out.append(s)  # noqa: E731
    try:
        os.environ["GPU_GATE_MB"] = "0"
        assert GG.wait_for_room(_used=lambda: 99999.0, _sleep=sleep, _now=clock, log=lg, status=None) == 0.0
        os.environ["GPU_GATE_MB"] = "12000"
        assert GG.wait_for_room(_used=lambda: 11600.0, _sleep=sleep, _now=clock, log=lg, status=None) == 0.0
        os.environ.pop("GPU_GATE_MB")
        os.environ["GPU_GATE_TIMEOUT_S"] = "180"
        now[0] = 0.0
        assert abs(GG.wait_for_room(_used=lambda: 11600.0, poll_s=60, _sleep=sleep, _now=clock, log=lg, status=None) - 180.0) < 1e-6
    finally:
        os.environ.pop("GPU_GATE_MB", None)
        os.environ.pop("GPU_GATE_TIMEOUT_S", None)
    with tempfile.TemporaryDirectory() as d:                                                                       # the waiting lines also go to the status file (a step's own output is shown only at its end)
        st = os.path.join(d, "s.status")
        now[0] = 0.0
        GG.wait_for_room(_used=lambda: 11600.0, poll_s=60, timeout_s=120, who="unit test", _sleep=sleep, _now=clock, log=lg, status=st)
        txt = open(st, encoding="utf-8").read()
        assert "unit test" in txt and "waiting" in txt and "going on" in txt, txt
        GG.wait_for_room(_used=lambda: 1.0, _sleep=sleep, _now=clock, log=lg, status=os.path.join(d, "no_such_dir", "x"))   # an unwritable status file never breaks the gate


def test_call_sites_gate_before_the_model():
    """the gate sits before the first thing that loads the model or builds the loader, in both commands (a deleted call would pass every other test)"""
    for fn, marker_after in (("jends_eval.py", "m, args = load_model(a.run, dev)"), ("jmaps_eval.py", "model.to(dev).eval()")):
        src = open(os.path.join(EVAL, fn), encoding="utf-8").read()
        i, j = src.index("gpu_gate.wait_for_room("), src.index(marker_after)
        assert 0 < i < j, (fn, i, j)
        assert 'if dev == "cuda"' in src[max(0, i - 400):i], fn
    # jends_eval.dump really calls it first: a recorder, a patched is_available and a load_model that raises stop the run right after the gate
    import torch
    import jends_eval as JEV
    calls = []
    old_av, old_wait, old_load = torch.cuda.is_available, GG.wait_for_room, JEV.load_model
    try:
        torch.cuda.is_available = lambda: True
        GG.wait_for_room = lambda **kw: calls.append(kw.get("who"))
        def boom(*a, **k):
            calls.append("load_model")
            raise RuntimeError("stop")
        JEV.load_model = boom
        try:
            JEV.dump(types.SimpleNamespace(run="e312_p_s0", thr=0.1, max=0))
        except RuntimeError:
            pass
        assert calls[:2] == ["jends_eval dump e312_p_s0", "load_model"], calls
    finally:
        torch.cuda.is_available, GG.wait_for_room, JEV.load_model = old_av, old_wait, old_load


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("test_gpu_gate ok")
