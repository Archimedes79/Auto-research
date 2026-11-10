import re


def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:90])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_adapter.py", [
    ('''    """(Source, Subject, Task) of a training run. split: 'val' | 'train' (overfitting check). prior > 0:''',
     '''    """(Source, Subject, Task) of a training run. split: the set to evaluate on (evaluation/bev_sets.py): 'val' | 'train' = the run's own lists (the train one is the
    overfitting check), '<split>.<part>' (small.val) = the lists of that file of splits/ put into the config before the data is made. prior > 0:'''),
    ('''    freeze_scenes(cfg)
    cfg.workers, cfg.seq_len, cfg.batch_size = 0, 1, 1''',
     '''    freeze_scenes(cfg)
    from evaluation import bev_sets
    cfg.train_scenes, cfg.val_scenes = bev_sets.lists(split, cfg.train_scenes, cfg.val_scenes)       # the run's own lists for val / train, the file's for small.val
    cfg.workers, cfg.seq_len, cfg.batch_size = 0, 1, 1'''),
    ('''    ds = train_ds if split == "train" else val_ds
    source = _make_source_class()''',
     '''    ds = train_ds if bev_sets.part_of(split) == "train" else val_ds
    source = _make_source_class()'''),
])

patch("evaluation/bev_selftest.py", [
    ('''bev_suite.tag_of(True, True, "zero_radar") == "_warm_train_ctl_zero_radar" and bev_suite.tag_of(False, False, "") == ""''',
     '''bev_suite.tag_of(True, "train", "zero_radar") == "_warm_train_ctl_zero_radar" and bev_suite.tag_of(False, "val", "") == "" and bev_suite.tag_of(False, "small.val", "") == "_small_val"'''),
])

patch("evaluation/bev_suite.py", [
    ('''def tag_of(warm: bool, on_train: bool, control: str) -> str:
    return ("_warm" if warm else "") + ("_train" if on_train else "") + (f"_ctl_{control}" if control else "")''',
     '''def tag_of(warm: bool, split: str, control: str) -> str:
    """the suffix of the outputs: warm, the set (bev_sets.tag: '' for val, '_train', '_small_val') and the control"""
    return ("_warm" if warm else "") + bev_sets.tag(split) + (f"_ctl_{control}" if control else "")'''),
    ('''    tag = tag_of(e.warm, a.split == "train", a.control)
    state = "carry" if e.warm else "reset"
    suite = make_suite(a, state=state)''',
     '''    tag = tag_of(e.warm, a.split, a.control)
    state = "carry" if e.warm else "reset"
    suite = make_suite(a, state=state)'''),
])
print("ok")
