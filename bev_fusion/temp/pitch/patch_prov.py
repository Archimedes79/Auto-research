p = "evaluation/bev_suite.py"
t = open(p, encoding="utf-8").read()
a1 = "def provenance_block(a, e, store, subj: dict, gt_sig, tag: str, state: str, start: dict = None) -> dict:"
b1 = '''def data_use(split: str, store, avail=None) -> dict:
    """which data an evaluation used, readable without the code: the split is the nuScenes norm (nuscenes.utils.splits.create_splits_scenes: train 700 /
    val 150 scenes); only the scenes whose features are cached can be run, so `available` is smaller than `official`; `scenes` are the ones the frames of
    this evaluation came from (their names, so another person can look them up); the test split of nuScenes has no public ground truth and is not used"""
    try:
        from nuscenes.utils.splits import create_splits_scenes
        official = set(create_splits_scenes().get(split, []))
    except Exception:                                    # no devkit: the counts stay unknown, the names stay
        official = set()
    used = sorted(store.scene_names())
    return {"split_rule": f"nuScenes official split '{split}' (nuscenes.utils.splits.create_splits_scenes), restricted to the scenes with cached features; "
                          "the nuScenes test split has no public ground truth and is not used",
            "scenes_official": len(official) or None, "scenes_available": len(avail) if avail is not None else None, "scenes_evaluated": len(used),
            "frames_evaluated": len(store.frames), "scenes_not_in_official_split": [s for s in used if official and s not in official],
            "scene_names": used}


def provenance_block(a, e, store, subj: dict, gt_sig, tag: str, state: str, start: dict = None, avail=None) -> dict:'''
assert a1 in t
t = t.replace(a1, b1)
a2 = "        ev, sub = pv.get(\"evaluator\", {}), pv.get(\"subject\", {})"
b2 = "        pv.setdefault(\"data\", {})[\"use\"] = data_use(a.split, store, avail)\n" + a2
assert a2 in t
t = t.replace(a2, b2)
a3 = "provenance_block(a, e, store, subj, gt_sig, tag, state, start=prov0)"
b3 = "provenance_block(a, e, store, subj, gt_sig, tag, state, start=prov0, avail=list(suite.source.scenes()))"
assert a3 in t
t = t.replace(a3, b3)
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("patched")
