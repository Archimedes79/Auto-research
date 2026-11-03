"""throw-away: third batch of edits of soa_maps_v8.py (radar entries missing from the cache are computed in memory, never written)"""
p = "temp/t16/soa_maps_v8.py"
s = open(p, encoding="utf-8").read()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:70])
    s = s.replace(old, new)


rep('''    n_src = {"cache": 0, "store": 0}
    miss = {"gt": [], "radar": []}''', '''    n_src = {"cache": 0, "store": 0}
    miss = {"gt": []}
    n_radar = {"cache": 0, "computed_not_written": 0}''')
rep('''        if not a.no_model and not os.path.exists(val_ds._radar_path(nusc.get("sample", t), val_ds.gt.cam(t))):
            miss["radar"].append(t)
    if miss["gt"] or miss["radar"]:
        sys.exit(f"STOP: no GT (cache{'' if a.all_dts else ' or store'}) for {len(miss['gt'])} frames, no radar cache entry for {len(miss['radar'])}; first: {(miss['gt'] + miss['radar'])[:3]}")
    print(f"GT of the {len(sel)} frames: {n_src['cache']} from the trainer's cache, {n_src['store']} from the stored GT {os.path.basename(store_dir)}", flush=True)''',
    '''        if not a.no_model:
            n_radar["cache" if os.path.exists(val_ds._radar_path(nusc.get("sample", t), val_ds.gt.cam(t))) else "computed_not_written"] += 1
    if miss["gt"]:
        sys.exit(f"STOP: no GT (cache{'' if a.all_dts else ' or store'}) for {len(miss['gt'])} frames; first: {miss['gt'][:3]}")
    print(f"GT of the {len(sel)} frames: {n_src['cache']} from the trainer's cache, {n_src['store']} from the stored GT {os.path.basename(store_dir)}", flush=True)
    if not a.no_model:
        print(f"radar input of the {len(sel)} frames: {n_radar['cache']} from the radar cache of the run's code (signature {val_ds._rsig}), {n_radar['computed_not_written']} "
              "not in the cache: computed from the raw sweeps in memory (what the cache would hold), never written", flush=True)''')
rep('''    orig_gt = val_ds._gt
''', '''    orig_gt = val_ds._gt
    FRD = type(val_ds).__mro__[1]                                        # FrontRadarDataset: its _radar is what fills the cache
    assert FRD.__name__ == "FrontRadarDataset", FRD
    orig_radar = val_ds._radar

    def radar_no_write(sample, cur_sd):
        """the cached entry, else the same function the cache is filled by, without writing the entry"""
        if os.path.exists(val_ds._radar_path(sample, cur_sd)):
            return orig_radar(sample, cur_sd)
        return FRD._radar(val_ds, sample, cur_sd)
    val_ds._radar = radar_no_write
    radar_same = {"checked": 0, "equal": 0}                              # the live computation against the cache entry of frames that have both
    if not a.no_model:
        for _, t in [r for r in sel if os.path.exists(val_ds._radar_path(nusc.get("sample", r[1]), val_ds.gt.cam(r[1])))][:4]:
            smp, cur_ = nusc.get("sample", t), val_ds.gt.cam(t)
            c, l_ = orig_radar(smp, cur_), FRD._radar(val_ds, smp, cur_)
            radar_same["checked"] += 1
            radar_same["equal"] += int(set(c) == set(l_) and all(torch.equal(c[k], l_[k]) for k in c))
        print(f"radar: live computation against the cache entry: {radar_same['equal']} of {radar_same['checked']} frames equal", flush=True)
''')
rep('"gt_store": os.path.relpath(store_dir, WS).replace("\\\\", "/"),', '"gt_store": os.path.relpath(store_dir, WS).replace("\\\\", "/"), "radar_sources": n_radar, "radar_live_vs_cache": radar_same,')
open(p, "w", encoding="utf-8").write(s)
print("ok")
