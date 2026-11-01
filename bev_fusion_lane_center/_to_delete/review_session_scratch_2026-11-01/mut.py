"""run G1-G3 of tests/test_velocity_hurdle.py against object_head.py with one source mutation (argv[1] = mutant name); prints PASS / FAIL"""
import os, sys, types, torch
torch.set_num_threads(2)
ROOT = os.getcwd()
sys.path[:0] = [os.path.join(ROOT, "training")]
MUT = {
 "none": [],
 "speed1.0": [("MOVING_SPEED = 0.5 ", "MOVING_SPEED = 1.0 ")],
 "speed0.3": [("MOVING_SPEED = 0.5 ", "MOVING_SPEED = 0.3 ")],
 "movmask_mv": [("losses[\"mov\"] = w_mov * (bce * m).sum() / m.sum().clamp(min=1)", "losses[\"mov\"] = w_mov * (bce * mv).sum() / mv.sum().clamp(min=1)")],
 "movmask_mb": [("losses[\"mov\"] = w_mov * (bce * m).sum() / m.sum().clamp(min=1)", "losses[\"mov\"] = w_mov * (bce * mb).sum() / mb.sum().clamp(min=1)")],
 "gate_logit0.5": [("out[\"mov\"][b, 0, ix, iy].item() < 0:", "out[\"mov\"][b, 0, ix, iy].item() < 0.5:")],
 "gate_sigmoid_wrong": [("out[\"mov\"][b, 0, ix, iy].item() < 0:", "torch.sigmoid(out[\"mov\"][b, 0, ix, iy]).item() < 0.3:")],
 "movtarget_all1": [("t[\"ind\"]), (mv > 0).float(), reduction", "t[\"ind\"]), (m > 0).float().expand_as(mv), reduction")],
}[sys.argv[1]]
src = open("training/object_head.py").read()
for a, b in MUT:
    assert a in src, a
    src = src.replace(a, b)
mod = types.ModuleType("object_head"); mod.__file__ = "training/object_head.py"
sys.modules["object_head"] = mod
exec(compile(src.split("\nif __name__")[0], "object_head_mut", "exec"), mod.__dict__)
test = open("tests/test_velocity_hurdle.py").read().split("# G4a")[0].replace("import train as tr  # noqa: E402", "")
try:
    exec(compile(test, "test_mut", "exec"), {"__name__": "t", "__file__": os.path.join(ROOT, "tests/test_velocity_hurdle.py")})
    print(sys.argv[1], "-> test PASSES (mutant not caught)" if MUT else "-> PASS")
except AssertionError as e:
    print(sys.argv[1], "-> FAIL (caught):", str(e)[:120])
