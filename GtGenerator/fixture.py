"""A synthetic nuScenes folder and a stand-in generator for the self-tests of the store, the index and the scene sets: no nuScenes needed.

    tables = tiny_tables(folder)          <folder>/v1.0-mini/scene.json + sample.json: 2 sequences x 4 keyframes -> {scene name: [sample tokens in time order]}
    tiny_sets(folder, small=[...], ...)   one <name>.txt per set (what sets.DIR points to during a test)
    ToyGenerator()                        what the index writer reads of a generator: signature() (the code signature of this source), grid, ss
    sets_at(folder)                       sets.DIR set to `folder` for the duration
"""
import contextlib
import json
import os

SCENES = ("scene-0061", "scene-0103")          # names of the set `small`, so the real set files can be read against the tiny tables
KEYFRAMES = 4


def tiny_tables(folder, version="v1.0-mini", scenes=SCENES, keyframes=KEYFRAMES):
    """the json tables the store, the sets and the key read (scene, sample) for `scenes`, each with `keyframes` linked samples"""
    tab = os.path.join(folder, version)
    os.makedirs(tab, exist_ok=True)
    scene_rows, sample_rows, out = [], [], {}
    for i, name in enumerate(scenes):
        toks = [f"{i:02d}{k:02d}" + "0" * 28 for k in range(keyframes)]
        out[name] = toks
        scene_rows.append({"token": f"s{i}", "name": name, "first_sample_token": toks[0], "last_sample_token": toks[-1], "nbr_samples": keyframes})
        sample_rows += [{"token": t, "scene_token": f"s{i}", "timestamp": 10 ** 6 * k,
                         "prev": toks[k - 1] if k else "", "next": toks[k + 1] if k + 1 < keyframes else ""} for k, t in enumerate(toks)]
    for name, rows in (("scene", scene_rows), ("sample", sample_rows)):
        with open(os.path.join(tab, name + ".json"), "w") as f:
            json.dump(rows, f)
    return out


def tiny_sets(folder, **sets):
    """<folder>/<name>.txt with the scene names of each keyword"""
    os.makedirs(folder, exist_ok=True)
    for name, scenes in sets.items():
        with open(os.path.join(folder, name + ".txt"), "w") as f:
            f.write("".join(s + "\n" for s in scenes))


@contextlib.contextmanager
def sets_at(folder):
    from . import sets
    old, sets.DIR = sets.DIR, folder
    try:
        yield
    finally:
        sets.DIR = old


class ToyGenerator:
    """the part of a generator the index writer and the job guard read: signature(), grid, ss"""

    def __init__(self, signature=None, ss=4):
        from .grid import GridSpec
        from .suite import code_signature
        self._signature = signature or code_signature()
        self.grid, self.ss = GridSpec(), ss

    def signature(self):
        return self._signature
