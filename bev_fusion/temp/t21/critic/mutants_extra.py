"""critic: extra mutants of the new code against core_checks() of tests/test_stream_temporal.py (its definitions only)"""
import inspect, os, sys, textwrap
ROOT = "/mnt/c/Auto-research/bev_fusion"
src = open(ROOT + "/tests/test_stream_temporal.py").read()
head = src.split("# ------------------------------------------------------------------------------------------------ A - D: baseline")[0]
g = {"__file__": ROOT + "/tests/test_stream_temporal.py", "__name__": "tt"}
exec(compile(head, "test_stream_temporal_head", "exec"), g)
M, core_checks = g["M"], g["core_checks"]
B = M.BEVFusionModel
unroll_src = textwrap.dedent(inspect.getsource(B._unroll))


def variant(old, new, name):
    assert old in unroll_src, old
    ns = dict(vars(M))
    exec(unroll_src.replace(old, new).replace("def _unroll", "def _unroll_mut"), ns)
    return name, ns["_unroll_mut"]


call = "block(frames[:, k], [] if fused is None else [(fused, move)], cam_state, move, obs)"
muts = [
    variant(call, "block(frames[:, k], [] if fused is None else [(fused, move)], cam_state, move, obs if k == 0 else None)", "write mask only at the newest frame"),
    variant(call, "block(frames[:, k], [] if fused is None else [(fused, move)], cam_state, move, None)", "no write mask at all in the unroll"),
    variant("move = as_chain(steps[k + 1][:n], frames)", "move = as_chain(steps[k + 1][-n:], frames)", "last n sweeps instead of the first n"),
    variant(call, "block(frames[:, k], [] if fused is None else [(fused.detach(), move)], cam_state, move, obs)", "fused detached inside the unroll"),
]
for name, fn in muts[:0]:
    orig = B.__dict__["_unroll"]
    B._unroll = fn
    try:
        res = core_checks()
        died = [k for k, v in res.items() if not v]
    except Exception as e:
        died = [f"raised {type(e).__name__}: {e}"]
    finally:
        B._unroll = orig
    print(f"{'KILLED  ' if died else 'SURVIVED'} {name}: {died[:2]}", flush=True)

# D check (cold stream step = window model with one frame) under the "no write mask" mutant
name, fn = muts[1]
orig = B.__dict__["_unroll"]
B._unroll = fn
try:
    w, m = g["build"](), g["build"](cam_buffer="recurrent")
    m.load_state_dict(w.state_dict())
    gp = g["gap"](g["run"](m, g["mk"]([2])), g["run"](w, g["mk"]([2])), buf=False)
    print("D check under 'no write mask': gap", gp, "->", "KILLED" if gp != 0 else "SURVIVED")
finally:
    B._unroll = orig
