p = "../bev_fusion/temp/pitch/patch_round4_results.py"
s = open(p, encoding="utf-8").read()
old = '("        self._write_manifest()\n        return rs\n", "        rs.save()\n        return rs\n")'
new = '("        rs._write_manifest()\n        return rs\n", "        rs.save()\n        return rs\n")'
assert s.count(old) == 1, s.count(old)
open(p, "w", encoding="utf-8").write(s.replace(old, new))
print("ok")
