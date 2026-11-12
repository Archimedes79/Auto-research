import os, subprocess, sys
src = open("evaluation/jmaneuver.py", encoding="utf-8").read()
muts = [("TOL 1.0", "BOX, TOL, ANG, NEED, JUNC = 50.0, 1.5,", "BOX, TOL, ANG, NEED, JUNC = 50.0, 1.0,"), ("TOL 2.5", "BOX, TOL, ANG, NEED, JUNC = 50.0, 1.5,", "BOX, TOL, ANG, NEED, JUNC = 50.0, 2.5,"),
        ("ANG 20", "1.5, 40.0, 0.9, 8.0", "1.5, 20.0, 0.9, 8.0"), ("ANG 90", "1.5, 40.0, 0.9, 8.0", "1.5, 90.0, 0.9, 8.0"),
        ("NEED 0.6", "1.5, 40.0, 0.9, 8.0", "1.5, 40.0, 0.6, 8.0"), ("NEED 1.0", "1.5, 40.0, 0.9, 8.0", "1.5, 40.0, 1.0, 8.0"),
        ("JUNC 1", "1.5, 40.0, 0.9, 8.0", "1.5, 40.0, 0.9, 1.0"), ("JUNC 40", "1.5, 40.0, 0.9, 8.0", "1.5, 40.0, 0.9, 40.0"),
        ("no direction test", "ok[i] = bool(((tang[nb] * run_tan[i]).sum(1) >= COS).any())", "ok[i] = True"),
        ("only the first 6 neighbours", "ok[i] = bool(((tang[nb] * run_tan[i]).sum(1) >= COS).any())", "ok[i] = bool(((tang[nb[:6]] * run_tan[i]).sum(1) >= COS).any())"),
        (">= instead of >", "[0] for", "[0] for")]
for name, old, new in muts[:-1]:
    assert old in src, name
    open("evaluation/_jman_mut.py", "w", encoding="utf-8").write(src.replace(old, new, 1))
    r = subprocess.run([sys.executable, "evaluation/_jman_mut.py", "selftest"], capture_output=True, text=True)
    print(f"{name:30s}", "CAUGHT" if r.returncode != 0 else "NOT CAUGHT")
os.remove("evaluation/_jman_mut.py")
