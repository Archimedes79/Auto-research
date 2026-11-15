def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))

# trainers: --raster
sub("training/train/jmaps_train.py", '    ap.add_argument("--steps", type=int, default=3000)\n', '    ap.add_argument("--raster", default="legacy", choices=("legacy", "exact"), help="E306: the polygon targets (zone, conflict core, stop lines, crossings) rasterised with the exact area rule instead of the legacy half-cell shifted fill")\n    ap.add_argument("--steps", type=int, default=3000)\n')
sub("training/train/jmaps_train.py", "    T.seed_all(a.seed)\n    LD.QData.DEFAULT_JUNCTION = True\n", "    T.seed_all(a.seed)\n    if a.raster == \"exact\":                                                                              # read at the call by `jgt.fill_poly`, inherited by the loader workers\n        os.environ[\"BFLC_RASTER\"] = \"exact\"\n    LD.QData.DEFAULT_JUNCTION = True\n")
sub("training/train/jends_train.py", '    ap.add_argument("--steps", type=int, default=3000)\n', '    ap.add_argument("--raster", default="legacy", choices=("legacy", "exact"), help="E306: exact polygon targets for the maps loss of --e2e (the pair head itself reads no raster target)")\n    ap.add_argument("--steps", type=int, default=3000)\n')
sub("training/train/jends_train.py", "    T.seed_all(a.seed)\n    LD.QData.DEFAULT_JUNCTION = True\n", "    T.seed_all(a.seed)\n    if a.raster == \"exact\":\n        os.environ[\"BFLC_RASTER\"] = \"exact\"\n    LD.QData.DEFAULT_JUNCTION = True\n")
# scorer: the run's own raster, --raster overrides
sub("evaluation/jmaps_eval.py", "    ds, dl = LD.loader(\"val_clean\", bs=8, workers=a.workers, shuffle=False, map_mode=\"exact\", pixels=False)\n    assert ds.missing == 0",
    "    raster = a.raster or args.get(\"raster\", \"legacy\")                                                      # the targets of the zone / core / paint IoU are rasterised as the run was trained (or as asked)\n    if raster == \"exact\":\n        os.environ[\"BFLC_RASTER\"] = \"exact\"\n    ds, dl = LD.loader(\"val_clean\", bs=8, workers=a.workers, shuffle=False, map_mode=\"exact\", pixels=False)\n    assert ds.missing == 0")
sub("evaluation/jmaps_eval.py", "res = dict(run=a.run, base=base_path, frames=n, args=args, lane_gt_signature", "res = dict(run=a.run, base=base_path, frames=n, args=args, raster=raster, lane_gt_signature")
sub("evaluation/jmaps_eval.py", '    s.add_argument("--max", type=int, default=0)\n', '    s.add_argument("--max", type=int, default=0)\n    s.add_argument("--raster", default=None, choices=("legacy", "exact"), help="score the zone / core / paint IoU against targets rasterised this way (default: as the run was trained)")\n')
print("ok")
