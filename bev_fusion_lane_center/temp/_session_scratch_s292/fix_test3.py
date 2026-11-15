p = "temp/_session_scratch_s292/new/test_mnms_layer.py"
s = open(p, encoding="utf-8").read()
a = s[s.index("        idx = np.asarray(fr[\"idx\"]).astype(int)\n"):s.index("        lg_grid = torch.full(")]
b = ('        e = np.asarray(fr["e"], float)\n'
     '        pos = np.asarray(fr["pos"], float)\n'
     '        # the dump keeps the 512 best candidates (`idx` is only their rank), not their cells: a node lies within 1 m of its cell centre, so its cell follows from its position\n'
     '        iy = np.clip(np.floor((pos[:, 0] + 50.0) / 2.0).astype(int), 0, H - 1)\n'
     '        ix = np.clip(np.floor((pos[:, 1] + 50.0) / 2.0).astype(int), 0, W - 1)\n'
     '        check(len(set(zip(iy.tolist(), ix.tolist()))) == len(iy), "two candidates fall into one cell: the grid cannot be rebuilt from the dump")\n')
s = s.replace(a, b)
open(p, "w", encoding="utf-8", newline="\n").write(s)
print("ok")
