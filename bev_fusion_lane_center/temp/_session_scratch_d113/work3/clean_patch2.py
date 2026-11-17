import os
p = os.path.join(os.environ["BFLC_ROOT"], "temp/_session_scratch_d113/work3/patch_cut.py")
s = open(p, encoding="utf-8").read()
bad = """    pieces of a frame''') if False else None
    pieces of a frame''') if "`gt(token)` -> the point arrays of the lane GT\n    pieces of a frame" in s else None
"""
assert bad in s
s = s.replace(bad, """sub('''`gt(token)` -> the point arrays of the lane GT
    pieces of a frame (default: the lane GT store of this checkout);''', '''`gt(token)` -> the lane GT
    pieces of a frame (the dicts with 'pts' and 'segments'; default: the lane GT store of this checkout);''')
""")
open(p, "w", encoding="utf-8").write(s)
print("ok")
