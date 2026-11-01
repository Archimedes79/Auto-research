"""one-off: replace import_closure ... manifest in autoresearch/tools/provenance.py by the reviewed version"""
p = "autoresearch/tools/provenance.py"
s = open(p, encoding="utf-8").read()
new = open("bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/new_bundle_block.py", encoding="utf-8").read()
i, j = s.index("def import_closure("), s.index("def selftest():")
s = s[:i] + new + s[j:]
# CLI: --force
a = '    ap.add_argument("--card")\n'
assert s.count(a) == 1
s = s.replace(a, a + '    ap.add_argument("--force", action="store_true", help="manifest: rebuild an existing provenance.json")\n')
a = "        b = manifest(a.args[0], a.run_dir)\n"
assert s.count(a) == 1
s = s.replace(a, "        b = manifest(a.args[0], a.run_dir, force=a.force)\n")
# repro: warn when the run is not final / the bundle is incomplete
a = '        L.append(f"manifest of the run: {pj} (made {prov.get(\'made\')})")\n'
assert s.count(a) == 1
b = a + (
    '        rs, bd = prov.get("run_status") or {}, prov.get("bundle") or {}\n'
    '        if rs and not rs.get("final"):\n'
    '            L.append(f"WARNING: the run is not final (status {rs.get(\'status\')}, exit {rs.get(\'exit\')}): its checkpoint is not a finished model")\n'
    '        for key, what in (("missing_roots", "root modules missing"), ("unparsed", "files that do not parse (their imports are unknown)"), ("imports_excluded", "the network imports the loop or a loader"),\n'
    '                          ("external_local", "repository code outside the bundle (its version is the run\'s GT version / tag)"), ("dynamic_imports", "files that import by name at run time")):\n'
    '            if bd.get(key):\n'
    '                L.append(f"bundle: {what}: {\', \'.join(bd[key][:6])}")\n'
)
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
