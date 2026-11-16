p = "temp/e291/mutate_e291.py"
s = open(p, encoding="utf-8").read()
def sub(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)
sub('''    shutil.copytree(STAGE, d)
    os.makedirs(f"{d}/training/grid", exist_ok=True)
    shutil.copy(f"{R}/{JPR}", f"{d}/{JPR}")
''', '''    guards = test[0] == "test_e291_guards.py"                                        # a launch script: the copy of the two scripts is mutated, the guards test runs them against the real tree
    if guards:
        os.makedirs(f"{d}/temp/e291")
        for n_ in ("start_arms.sh", "run_staged_gpu.sh"):
            shutil.copy(f"{R}/temp/e291/{n_}", f"{d}/temp/e291/{n_}")
    else:
        shutil.copytree(STAGE, d)
        os.makedirs(f"{d}/training/grid", exist_ok=True)
        shutil.copy(f"{R}/{JPR}", f"{d}/{JPR}")
''')
sub('''    env = dict(os.environ, BFLC_STAGE=d, E291_FAST="1")
    script''', '''    if guards:
        p = subprocess.run(["python3", f"{R}/temp/e291/test_e291_guards.py", "--start", f"{d}/temp/e291/start_arms.sh", "--wrapper", f"{d}/temp/e291/run_staged_gpu.sh"], capture_output=True, text=True, env=dict(os.environ, E291_ROOT=R))
        errs = [l for l in p.stderr.strip().splitlines() if "Error" in l]
        shutil.rmtree(d)
        return p.returncode != 0, (errs or [""])[-1][:160]
    env = dict(os.environ, BFLC_STAGE=d, E291_FAST="1")
    script''')
sub('''    # the staged hunks''', '''    ("p19_sc_counts_not_modulo", JPR, r"par\[key\]\[np\.arange\(len\(own\[key\]\)\) % k\]", "par[key][np.minimum(np.arange(len(own[key])), k - 1)]", JS, "SC: arm i takes the count of the partner's last arm instead of arm i modulo its number"),
    ("p20_sc_partner_none_not_unknown", JPR, r"if k else np\.full\(len\(own\[key\]\), UNK, np\.int64\)", "if k else own[key]", JS, "SC: a partner without arms leaves the own counts instead of unknown"),
    ("p21_sc_geometry_from_partner", JPR, r"            own\[key\] = par\[key\]([^\n]*)\n        return own\n", "            own[key] = par[key]\\1\n        return par\n", JS, "SC returns the partner's arms (geometry too)"),
    # the staged hunks''')
sub('''    ("s10_partner_not_other_frame"''', '''    ("s11_dump_folders_before_the_pass", "evaluation/jends_eval.py", r'        assert not os\.path\.exists\(d\), f"\{d\} exists: a read-out folder is never overwritten"', "        os.makedirs(d, exist_ok=True)", TS("dump"), "the read-out folders are made before the pass (a crash leaves them, the names are not checked first)"),
    ("s12_dump_no_max_suffix", "evaluation/jends_eval.py", r'\(f"_max\{a\.max\}" if a\.max else ""\)', '""', TS("dump"), "a --max dump takes the names of the real read-outs"),
    ("s13_partner_not_cached", "training/data/lane_gt_data.py", r"        if self\._pjg\[0\] == token:", "        if False:", TS("keys"), "(equivalent mutant: the cache only saves a GT load; must SURVIVE)"),
    ("g01_start_stage_conditional", "temp/e291/start_arms.sh", r'export BFLC_STAGE="\$\(pwd\)/temp/_session_scratch_e291/stage"', 'export BFLC_STAGE="${BFLC_STAGE:-$(pwd)/temp/_session_scratch_e291/stage}"', GD, "start_arms.sh takes the stage of the shell (E347's)"),
    ("g02_sha_check_off", "temp/e291/start_arms.sh", r'\[ -f "\$SHA" \] && sha256sum -c --quiet "\$SHA" \|\| \{', "true || {", GD, "the freeze check is skipped"),
    ("g03_runs_check_off", "temp/e291/start_arms.sh", r'if \[ -e "\$RUNS/e291_\$\{ARM\}_s\$S" \] && \[ "\$E291_RESUME" != "1" \]; then', "if false; then", GD, "an existing run folder is not refused"),
    ("g04_resume_ignored", "temp/e291/start_arms.sh", r' && \[ "\$E291_RESUME" != "1" \]; then', "; then", GD, "E291_RESUME=1 does not let a stopped run be resumed by hand"),
    ("g05_wrapper_takes_any_stage", "temp/e291/run_staged_gpu.sh", r'\|\| \{ echo "run_staged_gpu\.sh: BFLC_STAGE=\$BFLC_STAGE is not \$STAGE_OK \(another chat\'s stage\?\): REFUSED" >&2; exit 2; \}', "|| true", GD, "the GPU wrapper does not refuse another stage"),
    ("s10_partner_not_other_frame"''')
sub('''TS = lambda *sec:''', '''GD = ("test_e291_guards.py", [])
TS = lambda *sec:''')
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
