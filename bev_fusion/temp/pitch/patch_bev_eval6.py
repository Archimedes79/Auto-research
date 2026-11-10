def patch(path, pairs):
    s = open(path, encoding="utf-8").read()
    for old, new in pairs:
        assert s.count(old) == 1, (path, s.count(old), old[:110])
        s = s.replace(old, new)
    open(path, "w", encoding="utf-8").write(s)


patch("evaluation/bev_eval.py", [
    ('''def task(net: str, metrics=("maps", "boxes")):''', '''def free_source(net: str, of: str, workers: int = 2, control: str = ""):
    """the Source the Runner reads: the same set, frames and dataset keys as `source`, without the ground truth (bev_free.FreeSource); the Subject travels with it"""
    from evaluation.bev_free import free_source as make
    built = source(net, of, workers, control)
    free = make(built)
    built._bev_subject = None                                           # the free Source holds it now; the labelled one is only a reference reader from here on
    _SOURCES.pop(getattr(built, "_bev_key", None), None)
    return free


def free_subject(net: str, src, device: str):
    """the network without the validation loss (bev_free.FreeSubject over the BEVSubject `subject` hands out)"""
    from evaluation.bev_free import FreeSubject
    return FreeSubject(subject(net, src, device))


def task(net: str, metrics=("maps", "boxes")):'''),
])
print("ok")
