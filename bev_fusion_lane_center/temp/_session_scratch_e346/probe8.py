import os, glob
import e346_arm as E
import e346_lib as L
out = E.SMOKE + "/probe_g0.json"
orig = L.frame_record
# kill line: a g0 that does not score 1.000 -> exit 1, no card, a killed_ file next to it
L.frame_record = lambda arm, f, override=None: orig(arm, f, override=dict(f.gt, pts=[p + [1.5, 0.0] for p in f.gt["pts"]]))
rc = E.run_arm("g0", n=5, out=out, selftest=False)
print("kill line exit code", rc, "| card written:", os.path.exists(out), "| killed file:", os.path.exists(E.SMOKE + "/killed_probe_g0.json"))
L.frame_record = orig
# a --max card may not go to runs/eval
print("max into runs/eval exit code", E.main(["g0", "--max", "5", "--out", E.CARDS + "/e346_g0/val.json"]))
# a self-test failure: a harness that is wrong -> exit 1
orig_check = E.check_geometry_and_collisions
E.check_geometry_and_collisions = lambda: (_ for _ in ()).throw(AssertionError("planted"))
print("self-test failure exit code", E.main(["g1", "--max", "5"]))
E.check_geometry_and_collisions = orig_check
# an error (bad data) -> exit 2
E.QE.val_frames = lambda: 1 / 0
print("error exit code", E.main(["g1", "--max", "5", "--skip-selftest"]))
for f in glob.glob(E.SMOKE + "/*probe_g0.json"):
    os.remove(f)
print("exists after clean:", sorted(os.listdir(E.SMOKE)))
