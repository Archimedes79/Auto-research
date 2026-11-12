def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:90])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))
p = "training/grid/jgt.py"
sub(p, "    ii, jj = np.meshgrid(-half + (np.arange(n) + 0.5) * res, -half + (np.arange(n) + 0.5) * res, indexing=\"ij\")\n    for k, E in enumerate((jg[\"ent\"], jg[\"ext\"])):",
    "    sig = float(os.environ.get(\"BFLC_SIG\", SIG))                                                          # E307: the width of the end Gaussians (default 1.5 m; BFLC_SIG=1.0 for the crowded ends), read at the call, inherited by the loader workers\n    ii, jj = np.meshgrid(-half + (np.arange(n) + 0.5) * res, -half + (np.arange(n) + 0.5) * res, indexing=\"ij\")\n    for k, E in enumerate((jg[\"ent\"], jg[\"ext\"])):")
sub(p, "            h = np.exp(-d2 / (2 * SIG ** 2)).astype(np.float32)", "            h = np.exp(-d2 / (2 * sig ** 2)).astype(np.float32)")
sub(p, "            near = (d2 <= (2 * SIG) ** 2) & upd", "            near = (d2 <= (2 * sig) ** 2) & upd")
sub(p, "    zn, heat, head = fields(jg)\n    for k, EE in enumerate((jg[\"ent\"], jg[\"ext\"])):", "    zn, heat, head = fields(jg)\n    os.environ[\"BFLC_SIG\"] = \"1.0\"                                                                              # E307: a narrower Gaussian is lower one metre from the end and equal at the end\n    try:\n        _, heat_n, _ = fields(jg)\n    finally:\n        os.environ.pop(\"BFLC_SIG\", None)\n    e0 = jg[\"ent\"][0]\n    ci, cj = int((e0[0] + 50.0) // 1.0), int((e0[1] + 50.0) // 1.0)\n    assert heat_n[0][ci, cj] <= heat[0][ci, cj] + 1e-6 and heat_n[0][ci, min(cj + 2, 99)] < heat[0][ci, min(cj + 2, 99)], (heat_n[0][ci, cj], heat[0][ci, cj])\n    for k, EE in enumerate((jg[\"ent\"], jg[\"ext\"])):")
p = "training/train/jmaps_train.py"
sub(p, '    ap.add_argument("--steps", type=int, default=3000)\n', '    ap.add_argument("--sigma", type=float, default=1.5, help="E307: the width (m) of the Gaussian entry / exit heat targets (1.5 = E286)")\n    ap.add_argument("--steps", type=int, default=3000)\n')
sub(p, "    if a.raster == \"exact\":", "    if a.sigma != 1.5:\n        os.environ[\"BFLC_SIG\"] = str(a.sigma)\n    if a.raster == \"exact\":")
print("ok")
