p = "temp/e291/mutate_e291.py"
L = open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith('    ("p21_sc_geometry_from_partner"'))
j = next(k for k in range(i, len(L)) if L[k].startswith('", JS, "SC returns'))
bs = chr(92)
L[i:j + 1] = ['    ("p21_sc_geometry_from_partner", JPR, r"        return own' + bs + 'n    if spec' + bs + '.mode == ' + bs + '"S' + bs + '":", "        return par' + bs + 'n    if spec.mode == ' + bs + '"S' + bs + '":", JS, "SC returns the partner' + "'" + 's arms (geometry too)"),']
s = "\n".join(L)
a = s.index('    ("g05_wrapper_takes_any_stage"')
b = s.index("\n", a)
s = s[:a] + '    ("g05_wrapper_takes_any_stage", "temp/e291/run_staged_gpu.sh", r"' + bs + '|' + bs + '| ' + bs + '{ echo ' + bs + '"run_staged_gpu' + bs + '.sh: BFLC_STAGE=' + bs + '$BFLC_STAGE is not ' + bs + '$STAGE_OK[^' + bs + 'n]*exit 2; ' + bs + '}(?=' + bs + 'n  export BFLC_STAGE=)", "|| true", GD, "the GPU wrapper does not refuse another stage"),' + s[b:]
# s08 -> the assert removed alone (the folder is made at the end: FileExistsError after the pass, not the assertion before it)
a = s.index('    ("s08_dump_overwrites"')
b = s.index("\n", a)
s = s[:a] + '    ("s08_dump_no_name_check", "evaluation/jends_eval.py", r' + "'" + '        assert not os' + bs + '.path' + bs + '.exists' + bs + '(d' + bs + '), f"' + bs + '{d' + bs + '} exists: a read-out folder is never overwritten"' + "'" + ', "        pass", TS("dump"), "the names of the read-out folders are not checked before the pass"),' + s[b:]
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
