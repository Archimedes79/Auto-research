import io, subprocess, shutil
P = "temp/_session_scratch_e358/stage/jwire_head.py"
B = "temp/_session_scratch_e358/jwire_head.backup_e363"
src = io.open(B, encoding="utf-8").read()
M = [("source not compared", "assert src == hp, f", "assert True or src == hp, f"),
     ("wrong key read", 'json.load(open(pv)).get("source_pair_run")', 'json.load(open(pv)).get("pair_run")'),
     ("provenance path wrong", '"runs", run, "provenance.json"', '"runs", run, "provenance.jsn"'),
     ("derived check applied also when equal (inverted)", "    if hp != run:  ", "    if hp == run:  ")]
for name, a, b in M:
    assert src.count(a) == 1, (name, src.count(a))
    io.open(P, "w", encoding="utf-8").write(src.replace(a, b))
    r = subprocess.run(["bash", "temp/e358/run_scratch.sh", "temp/_session_scratch_e358/test_staged.py"], capture_output=True, text=True)
    print(name, "KILLED" if r.returncode != 0 else "SURVIVED")
shutil.copyfile(B, P)
print("restored")
