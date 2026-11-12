def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert old in s, (path, old[:80])
    open(path, "w", encoding="utf-8").write(s.replace(old, new, count))
sub("evaluation/jmaps_eval.py", '    out_path = os.path.join(run_dir, "jmaps_val.json")', '    out_path = os.path.join(run_dir, "jmaps_val.json" if a.raster is None else f"jmaps_val_{a.raster}.json")                  # a score on re-rasterised targets never overwrites the run\'s own file')
sub("evaluation/jmaps_eval.py", "-> runs/<run>/jmaps_val.json + a printed table", "-> runs/<run>/jmaps_val.json + a printed table (`--raster exact|legacy`: the zone / core / paint targets rasterised that way, written to jmaps_val_<raster>.json)")
sub("experiments/E306_exact-polygon-targets/experiment.md", "(the control file is written next to the E286 one as `jmaps_val.json`: run it with a copy of the run folder or name the output; see the scorer's `--out` if present)", "(written next to the E286 result as `jmaps_val_exact.json`, the run's own `jmaps_val.json` is never overwritten)")
print("ok")
