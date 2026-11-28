"""Tests of evaluation/bev_timing.py (its former self-test).

    python tests/test_bev_timing.py      (from the problem folder, with its module path)
"""
from __future__ import annotations
import json
import os
import subprocess
import time

import torch

from evaluation import bev_adapter as A
from evaluation import bev_timing
from evaluation.bev_timing import OWN_GPU_SCRIPTS, busy_processes, gpu_scripts, gpu_state, holds_gpu, is_idle, run_timing, settled_state, time_modules, train_split


def selftest():
    import torch.nn as nn

    class Sleep(nn.Module):
        def __init__(self, ms):
            super().__init__()
            self.ms = ms

        def forward(self, x):
            time.sleep(self.ms / 1e3)
            return x + 1

    class Trunk(nn.Module):
        def features(self, x):
            time.sleep(0.015)
            return x + 1

    class Outer(nn.Module):                                             # a module that calls another probed module (like a head calling a shared mover)
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            time.sleep(0.005)
            return self.inner(x)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lift, self.radar, self.sem_head, self.trunk = Sleep(10), Sleep(20), Sleep(5), Trunk()
            self.mover = Sleep(20)
            self.wrapper = Outer(self.mover)                            # `mover` is a child of the model too: 20 ms inside a 5 ms caller

        def forward(self, x):
            x = self.trunk.features(self.radar(self.lift(x)))
            for _ in range(3):                                          # a head called once per dt
                self.sem_head(x)
            return self.wrapper(x)
    m = Toy()
    plain = m.trunk.features
    reference = m(torch.zeros(1))
    res = time_modules(m, lambda: m(torch.zeros(1)), "cpu", n=8, warmup=2)
    pf, pc, calls = res["per_frame_ms"], res["per_call_ms"], res["calls"]
    near = lambda v, want: want <= v < want * 1.6 + 4        # noqa: E731  (a sleep never returns early; the overhead is small but a loaded machine adds some)
    assert near(pc["lift"], 10) and near(pc["radar"], 20) and near(pc["trunk"], 15) and near(pc["sem_head"], 5), pc
    assert calls["sem_head"] == 3 and calls["lift"] == 1 and calls["trunk"] == 1, calls
    assert near(pf["sem_head"], 15) and abs(pf["sem_head"] - 3 * pc["sem_head"]) < 4, "per frame = the sum of the calls"
    assert near(pc["wrapper"], 5) and pc["wrapper"] < 12 and near(pc["mover"], 20) and pc["mover"] < 30, ("exclusive: the caller keeps only its own 5 ms", pc)
    assert pf["radar"] > pf["trunk"] > pf["lift"] and abs(res["unattributed_ms"]) < 0.25 * res["total_ms"] + 5, (pf, res)
    assert abs(res["total_ms"] - sum(pf.values()) - res["unattributed_ms"]) < 1e-6
    assert m.trunk.features == plain and "features" not in m.trunk.__dict__ and not any(mod._forward_hooks or mod._forward_pre_hooks for mod in m.children()), "hooks and wrapper removed"
    assert torch.equal(m(torch.zeros(1)), reference), "the timing hooks do not change the output"
    # the training split: a loss whose forward takes 6 ms and whose backward takes 9 ms

    class Slow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            time.sleep(0.006)
            return x * 1.0

        @staticmethod
        def backward(ctx, g):
            time.sleep(0.009)
            return g
    w = torch.ones(1, requires_grad=True)
    ts = train_split(lambda: Slow.apply(w).sum(), lambda: setattr(w, "grad", None), "cpu", n=5, warmup=1)
    assert near(ts["forward_loss"], 6) and near(ts["backward"], 9) and ts["forward_loss_min_max"][0] <= ts["forward_loss"] <= ts["forward_loss_min_max"][1], ts
    # the idle rule: GPU utilisation, memory, processes (nvidia-smi and ps are read; our own pid is not another user), unknown is busy; a CPU run is idle
    ok = {"util_percent": 2.0, "mem_used_mb": 1300.0, "other_processes": [], "busy_processes": []}
    assert is_idle(ok, "cuda") and is_idle({**ok, "util_percent": 99}, "cpu")
    for bad in ({"util_percent": 99.0}, {"util_percent": None}, {"mem_used_mb": 11800.0}, {"mem_used_mb": None}, {"other_processes": ["1, 5000 MiB"]}, {"busy_processes": ["7 python train.py"]}):
        assert not is_idle({**ok, **bad}, "cuda"), bad
    assert is_idle({**ok, "mem_used_mb": 11800.0}, "cuda", check_memory=False), "after our own model is loaded the memory says nothing"
    # nvidia-smi under WSL2 lists the process that holds a CUDA context with its own pid (seen 2026-11-03: the second check, taken after the model load, refused
    # an idle GPU
    # because it counted this very process): the listing of gpu_state drops the own pid and ONLY that one (the parent is no GPU user either, but guessing is not
    # our business)
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] != "nvidia-smi":
            return real_run(cmd, **kw)
        text = "5, 1000\n" if cmd[1].startswith("--query-gpu") else f"{os.getpid()}, [N/A]\n4242, [N/A]\n{os.getppid()}, [N/A]\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")
    subprocess.run = fake_run
    try:
        listed = gpu_state(samples=1)["other_processes"]
    finally:
        subprocess.run = real_run
    assert listed == ["4242, [N/A]", f"{os.getppid()}, [N/A]"], ("only the own pid is not another GPU user", listed)
    # the second look is taken after our own work has left the GPU: nvidia-smi's utilisation covers the last second, and the tail of the model load read 12 - 33
    # % (seen 2026-11-03,
    # nothing else running, 0 % half a second later), which refused an idle GPU; a CPU run waits for nothing, and so does a "cuda" request that fell back to the
    # CPU (no CUDA GPU:
    # torch.cuda.synchronize would raise where the old code reached the warning path of run_timing)
    calls = []
    saved = (vars(bev_timing)["gpu_state"], torch.cuda.synchronize, torch.cuda.is_available, time.sleep)
    vars(bev_timing)["gpu_state"], torch.cuda.synchronize, time.sleep = (lambda samples=3, gap=0.5: calls.append(f"look {samples}") or {}), (lambda *a: calls.append("sync")), (lambda s: calls.append(f"wait {s}"))
    try:
        for available, device in ((True, "cuda"), (True, "cpu"), (False, "cuda")):
            torch.cuda.is_available = lambda available=available: available
            settled_state(device, samples=2)
    finally:
        vars(bev_timing)["gpu_state"], torch.cuda.synchronize, torch.cuda.is_available, time.sleep = saved
    assert calls == ["sync", "wait 1.5", "look 2", "look 2", "look 2"], calls
    # run_timing: refuses a busy GPU at the first look (before the model is built) and at the second one (taken through settled_state), goes on when both are
    # idle or with --force
    from types import SimpleNamespace
    busy, seen = {**ok, "util_percent": 50.0}, []

    def timing_with(first, second, **kw):
        """what run_timing does with these two looks (fakes for the model build): the refusal text, or 'went on' (the fake model has no frames: it stops
        there)"""
        seen.clear()
        subj = SimpleNamespace(model=None, loss=None, cfg=SimpleNamespace(amp="off"), device="cuda:0", step=0)
        saved = (vars(bev_timing)["gpu_state"], vars(bev_timing)["settled_state"], A.build)
        vars(bev_timing)["gpu_state"], vars(bev_timing)["settled_state"] = (lambda samples=3, gap=0.5: seen.append("first") or first), (lambda device, samples=2: seen.append(f"second {device}") or second)
        A.build = lambda run, **k: seen.append("build") or (None, subj, None)
        try:
            run_timing("runs/x", device="cuda", **kw)
        except SystemExit as e:
            return str(e)
        except Exception:                                                  # noqa: BLE001
            return "went on"
        finally:
            vars(bev_timing)["gpu_state"], vars(bev_timing)["settled_state"], A.build = saved
    assert timing_with(busy, ok).startswith("refused: the GPU is not idle") and seen == ["first"], seen
    assert timing_with(ok, busy).startswith("refused before the timed loop") and seen == ["first", "build", "second cuda"], seen
    assert timing_with(ok, ok) == "went on" and seen == ["first", "build", "second cuda"], seen
    assert timing_with(ok, busy, force=True) == "went on" and timing_with(busy, ok, force=True) == "went on", "--force times a busy GPU (and marks the result not idle)"
    # the GPU steps of the problems: a step with "gpus" >= 1 names its scripts, a step without, the core's evaluation step and a broken problem.json do not (the
    # latter is an error)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for name, steps in (("a", {"train": {"run": ["{python}", "training/train.py", "{config}"], "gpus": 1}, "prepare": {"run": ["{python}", "gt/generate.py", "{config}"]}}),
                            ("b", {"stage1": {"run": ["{python}", "training/train/lane_train.py", "stage1", "{config}"], "gpus": 1}, "evaluate": {"run": "evaluate", "gpus": 1}})):
            os.makedirs(os.path.join(tmp, name))
            json.dump({"steps": steps}, open(os.path.join(tmp, name, "problem.json"), "w"))
        scripts = gpu_scripts(tmp)
        assert scripts == {**OWN_GPU_SCRIPTS, "train.py": "cuda", "lane_train.py": "cuda"}, scripts
        open(os.path.join(tmp, "b", "problem.json"), "w").write("{broken")
        try:
            gpu_scripts(tmp)
            raise AssertionError("a broken problem.json was read as no GPU step")
        except ValueError as e:
            assert "problem.json" in str(e), e
    assert gpu_scripts().get("train.py") == "cuda", "this problem's own train step (problem.json: gpus 1) is a GPU script"
    ps = ("  PID COMMAND\n  101 /home/ml/miniconda3/envs/gcar/bin/python training/train.py /runs/bev_fusion/E27.a_s0/config.json\n"
          "  102 python evaluation/bev_timing.py runs/x\n  103 bash -c sleep 5\n  104 python other.py\n")
    assert [x.split()[0] for x in busy_processes(ps, {102}, scripts)] == ["101"] and busy_processes(ps, {101, 102}, scripts) == [], "ps: trainers / evaluations, not ourselves"
    # a run's worker counts, also while it waits for its GPU slot (its step would start under the timing); the commands of the core that start or show runs do
    # not
    research = ("  PID COMMAND\n  201 /home/ml/miniconda3/envs/gcar/bin/python -m research worker bev_fusion E27.a_s0\n  202 python3 -m research status bev_fusion\n"
                "  203 python3 -m research start bev_fusion E27\n  204 python -m research.evaluate bev_fusion E27.a_s0\n"
                "  205 python training/train/lane_train.py stage1 /runs/lane_center/E400.a_s0/config.json\n  206 python gt/generate.py make --config cfg\n  207 python -c import train.py\n")
    assert [x.split()[0] for x in busy_processes(research, set(), scripts)] == ["201", "204", "205"], busy_processes(research, set(), scripts)
    for line, want in (("python3.12 -u /runs/_code/abc/problems/bev_fusion/training/train.py cfg.json", True), ("python -W ignore training/train.py cfg.json", True),
                       ("python -X dev -u train.py cfg", True), ("env CUDA_VISIBLE_DEVICES=0 python train.py cfg", True), ("python -m evaluation.bev_timing runs/x", True),
                       ("python -m evaluation.bev_timing runs/x --device=cpu", False), ("python evaluation/bev_timing.py runs/x", True), ("python evaluation/bev_modules.py runs/x", False),
                       ("python evaluation/bev_modules.py runs/x --device cuda", True), ("python net/prep/cache_bevfusion_features.py", True),
                       ("python -m research worker toy E4.m_s0", True), ("python -m research.evaluate lane_center E400.a_s0", True), ("python -m research status", False),
                       ("python -c train.py", False), ("python -m json.tool train.py", False), ("bash -c python training/train.py cfg", False), ("python other.py train.py", False)):
        assert holds_gpu(line, scripts) == want, (line, want)
    print("bev_timing selftest ok: per call / per frame / exclusive milliseconds of toy modules with known sleeps, hooks removed and output unchanged, training split, idle rule")


if __name__ == "__main__":
    selftest()
