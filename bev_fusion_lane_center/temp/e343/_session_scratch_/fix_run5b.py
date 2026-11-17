import os, re
R = os.environ["BFLC_ROOT"]
p = os.path.join(R, "temp", "e343", "_session_scratch_", "run_mut5.py")
s = open(p, encoding="utf-8").read()
a = s.index("names = [")
b = s.index("part = int(sys.argv[1])")
names = ["pin run prefix not checked", "pin rehearsal ok not checked", "pin drift not checked", "pin digest of file list not checked", "pin card digest ignored", "pin environment record not checked",
         "environment compare off", "environment_diff versions ignored", "undigested not computed", "undigested includes preloaded", "repo_modules temp kept", "repo_modules GtGenerator kept", "repo_modules relative kept",
         "gpu gate removed (dump_b)", "gpu gate removed (dump_ends)", "gpu gate always (dump_b)", "gpu gate always (dump_ends)",
         "TEST artefact rule off", "precheck rows not validated", "early control off", "early counts scored again",
         "TEST boot not fixed", "TEST seeds not fixed", "TEST b_dump not required", "TEST control_ends not required", "TEST control_card not required", "TEST scenes not checked",
         "sidecar not first (dump_b)", "sidecar not first (dump_ends)", "pickle overwrite allowed (dump_b)", "pickle overwrite allowed (dump_ends)", "check_seed seed optional"]
m = open(os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py"), encoding="utf-8").read()
for n in names:
    assert f' ("{n}",' in m, n
s = s[:a] + "names = " + repr(names) + "\n" + s[b:]
open(p, "w", encoding="utf-8").write(s)
print(len(names), "ok")
