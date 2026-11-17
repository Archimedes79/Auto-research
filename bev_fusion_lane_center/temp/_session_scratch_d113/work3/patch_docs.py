import os
R = os.environ["BFLC_ROOT"]


def sub_file(path, pairs):
    p = os.path.join(R, path)
    s = open(p, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(p, "w", encoding="utf-8").write(s)


sub_file("training/data/lane_map_data.py", [
    ("# the TEST guard (VAL rows = data/splits.json['val']), the leak check on every built table, the truncation counts of VAL and TRAIN (exact and error map)", "# the TEST guards (VAL rows = data/splits.json['val']; TRAIN rows = splits['train'], no TRAIN sample token a VAL / TEST frame's), the leak check on every built table, the truncation counts of VAL and TRAIN (exact and error map)"),
])
sub_file("training/grid/README.md", [
    ("K = 512 x M = 704 below 3 GB", "K = 512 x M = 704 below 3 GB, the bf16-autocast logits summed in float32"),
])
sub_file("training/grid/test_jmap_train.py", [
    ("the real loader check (`real`, needs the cache and the lane GT store): the batch of arm z is `lane_ptok_data`'s, the maps are those of `lane_map_data`, `train()` hands the right arguments to the loader.", "the real loader check (`real`, needs the cache and the lane GT store): the batch of arm z is `lane_ptok_data`'s, the maps are those of `lane_map_data`, `train()` hands the right arguments to the loader (also for `--map-mode other`, the capacity control). More: the maps are moved to the head's device (a `meta`-device probe), the defaults of `train()`, the TRAIN sample dump, `params` / `time` and the strict `load_model`."),
])
print("docs ok")
