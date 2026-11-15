import os
R = os.environ["BFLC_ROOT"]
T = os.path.join(R, "evaluation", "tests", "test_jtest_eval.py")
s = open(T, encoding="utf-8").read()
old = '''"runs/eval/e324s_train/never.json", "never.txt"):'''
new = '''"runs/eval/e324s_train/never.json", "temp/e3430/never.txt", "runs/eval/e312_val2/never.json", "never.txt"):'''
assert s.count(old) == 1
open(T, "w", encoding="utf-8").write(s.replace(old, new))

M = os.path.join(R, "temp", "e343", "_session_scratch_", "mutate.py")
t = open(M, encoding="utf-8").read()
anchor = '''# glue
 ("glue attach dropped"'''
rows = r'''
 # round 3
 ("sidecar old head refused", 'got = json.load(f).get("provenance")', 'got = json.load(f).get("provenance") or {"run": None}'),
 ("precheck no flagged/default check", 'for path, run, what, required in dumps:', 'for path, run, what, required in []:'),
 ("precheck default without sidecar ok", '(required or sidecar_provenance(path) is not None)', '(sidecar_provenance(path) is not None)'),
 ("message interrupted", 'made by other code OR interrupted between the pickle and its sidecar', 'made by other code'),
 ("precheck ctl_b always named", '([i["ctl_ends"]] + ([i["ctl_b"]] if i["ctl_b"] != i["b"] else []) if setup["arms"] and i["ctl_ends"] else [])', '([i["ctl_ends"], i["ctl_b"]] if setup["arms"] and i["ctl_ends"] else [])'),
 ("code dirs no subfolders", 'dn[:] = [] if depth >= 4 else sorted(d for d in dn if d != "__pycache__" and "_session_scratch_" not in d)', 'dn[:] = []'),
 ("code lazy never followed", 'for n in (ast.walk(tree) if lazy else _executed(tree)):', 'for n in _executed(tree):'),
 ("code lazy followed in autoresearch", 'lazy=os.path.relpath(p, repo).split(os.sep)[0] != "autoresearch"', 'lazy=True'),
 ("code scratch walked", 'sorted(d for d in dn if d != "__pycache__" and "_session_scratch_" not in d)', 'sorted(d for d in dn if d != "__pycache__")'),
 ("code skip evaluate off", 'CODE_SKIP = ("evaluate", "GtGenerator")', 'CODE_SKIP = ("GtGenerator",)'),
 ("code digest no stamp size", 'return path, st.st_mtime_ns, st.st_size', 'return path, st.st_mtime_ns, 0'),
 ("provenance line removed dump_b", '    prov = provenance(run, device)                                                                     # the code and the checkpoint as they are when the dump STARTS (not after 10 - 25 minutes)\n', ''),
 ("provenance line removed dump_ends", '    prov = provenance(run, device)                                                                     # at the start, as in dump_b\n', ''),
 ("expect ignored", 'if a.expect_code_digest and not (', 'if False and not ('),
 ("expect short prefix ok", 'len(a.expect_code_digest) >= 16 and digest.startswith(a.expect_code_digest)', 'digest.startswith(a.expect_code_digest)'),
 ("expect digest not printed", 'print(f"code digest: {digest} ({len(code0)} files)", flush=True)', 'pass'),
 ("expect card names none", 'changed = sorted(k for k in set(ref) | set(code0) if ref.get(k) != code0.get(k))', 'changed = []'),
 ("atomic no allow-list", '    if not is_writable(path):', '    if False:'),
 ("writable too broad", 'WRITABLE = ("temp/e343", ', 'WRITABLE = ("temp", '),
 ("writable prefix no sep", 'real.startswith(r.casefold() + os.sep) for r in roots', 'real.startswith(r.casefold()) for r in roots'),
 ("writable no casefold", 'real = os.path.realpath(path).casefold()\n    roots = [os.path.realpath(os.path.join(ROOT, p)) for p in WRITABLE]', 'real = os.path.realpath(path)\n    roots = [os.path.realpath(os.path.join(ROOT, p)) for p in WRITABLE]'),
 ("precheck seed unchecked flagged", 'if os.path.exists(path) and (required or sidecar_provenance(path) is not None):', 'if os.path.exists(path) and required:'),
'''
assert t.count(anchor) == 1
t = t.replace(anchor, rows + anchor)
open(M, "w", encoding="utf-8").write(t)
print("ok")
