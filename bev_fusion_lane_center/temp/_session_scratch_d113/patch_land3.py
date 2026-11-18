import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp", "landing", "land_patch.py")
s = open(p, encoding="utf-8").read()
a = '''    print(f"applied: {len(changed)} files")'''
assert s.count(a) == 1
s = s.replace(a, '''    with open(os.path.join(root, "temp", "landing", "changed.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sorted(changed)) + "\n")
    print(f"applied: {len(changed)} files (list in temp/landing/changed.txt)")''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
