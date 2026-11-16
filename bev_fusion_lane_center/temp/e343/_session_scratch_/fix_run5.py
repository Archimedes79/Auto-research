import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "e343", "_session_scratch_", "run_mut5.py")
s = open(p, encoding="utf-8").read()
a = s.index("names = [")
b = s.index("part = int(sys.argv[1])")
new = 'names = ["pin drift not checked", "pin digest of file list not checked", "pin card digest ignored", "environment compare off", "undigested not computed", "repo_modules temp kept", "gpu gate removed (dump_b)", "gpu gate always (dump_ends)", "TEST artefact rule off", "precheck rows not validated", "early control off", "early counts scored again", "TEST boot not fixed", "TEST control_card not required", "TEST scenes not checked", "sidecar not first (dump_b)", "pickle overwrite allowed (dump_ends)", "check_seed seed optional", "cmd_all no gate", "main dump no gate", "e321_changed empty", "pin run prefix not checked", "pin rehearsal ok not checked", "TEST seeds not fixed"]\n'
open(p, "w", encoding="utf-8").write(s[:a] + new + s[b:])
