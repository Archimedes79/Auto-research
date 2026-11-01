p = "bev_fusion_lane_center/temp/review_session_scratch_2026-11-01/splice_closure.py"
s = open(p, encoding="utf-8").read()
old = '''sub(\'\'\'("model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files, hash\'\'\', \'\'\'("model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files + {len(_get(prov, ('bundle', 'repo_files')) or [])} repository files, hash\'\'\')'''
new = '''sub(\'\'\'(f"model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files, hash\'\'\', \'\'\'(f"model bundle (network + losses): {len(_get(prov, ('bundle', 'files')) or [])} files + {len(_get(prov, ('bundle', 'repo_files')) or [])} repository files, hash\'\'\')
sub(\'\'\'"model bundle (network + losses): 8 files" in buf.getvalue()\'\'\', \'\'\'"model bundle (network + losses): 8 files + 0 repository files" in buf.getvalue()\'\'\')'''
assert s.count(old) == 1, s.count(old)
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("ok")
