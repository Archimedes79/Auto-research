"""The scene lists of the training splits (`../splits/<name>.json`) and what a run records of them. Standard library only.

    python training/splits.py resolve                  stdin: a config (json) -> stdout: the config with the lists of its "split" filled in
                                                       (`trainsuite start` calls it: project.json "trainsuite" -> "resolve")
    python training/splits.py names NAME               the scene names of a split, one per line (train, then val)
    python training/splits.py manifest RUN_DIR [--card records/runs/<id>/run.json]
                                                       writes RUN_DIR/manifest.json: the split, GT, code and card of a run, read from the files it wrote
    python training/splits.py split-json RUN_DIR OUT_DIR
                                                       writes OUT_DIR/split.json (the scene lists of the run; `package.py` puts it into results/<id>/)

A config says `"split": "large"` instead of carrying 500 scene names. `resolve` replaces the key by `train_scenes` / `val_scenes` (and the comment key `_split`, which
`train.read_config` ignores), so the config the trainer reads, and copies into `runs/<id>/config.json`, holds the exact lists: the run is pinned to them, not to the file.
A config without `"split"` is returned as it is (old configs with inline lists keep working); `"split": null` drops the key and keeps whatever the config has; a split and inline
lists together are refused (which one would win?). `train.py` itself does not know the key: a config with `split` that was not resolved stops there with "unknown key".
"""
import argparse
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPLITS = os.path.join(os.path.dirname(HERE), "splits")             # bev_fusion/splits/
LISTS = ("train_scenes", "val_scenes")                              # the config keys the lists go to


def load(name, splits_dir=SPLITS):
    """{"about", "gt", "train", "val"} of the split file <name>.json; ValueError for an unknown name or lists that overlap or repeat a scene"""
    if not re.fullmatch(r"[a-z0-9_]+", str(name)):
        raise ValueError(f"split {name!r}: lower-case letters, digits and _ only")
    path = os.path.join(splits_dir, name + ".json")
    if not os.path.isfile(path):
        have = sorted(f[:-5] for f in os.listdir(splits_dir) if f.endswith(".json")) if os.path.isdir(splits_dir) else []
        raise ValueError(f"no split {name!r} in {splits_dir} (there are: {', '.join(have) or 'none'})")
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    train, val = d.get("train"), d.get("val")
    if not (isinstance(train, list) and isinstance(val, list) and train and val):
        raise ValueError(f"{path}: needs the non-empty lists train and val")
    if len(set(train)) != len(train) or len(set(val)) != len(val) or set(train) & set(val):
        raise ValueError(f"{path}: a scene is listed twice or in train and val")
    return d


def resolve(cfg, splits_dir=SPLITS):
    """cfg with the lists of its "split" filled in (a copy); a config without the key is returned as it is"""
    if "split" not in cfg:
        return cfg
    out = {k: v for k, v in cfg.items() if k != "split"}
    name = cfg["split"]
    if name is None:                                                # "split": null = no split, the config keeps its own lists (e.g. a base with a split + a preset that brings a small set)
        return out
    if not isinstance(name, str):
        raise ValueError(f"split must be a split name (a string) or null, got {name!r}")
    if any(out.get(k) for k in LISTS):
        raise ValueError(f"split {name!r} and inline train_scenes / val_scenes in one config: keep one. Base with lists + a split: --set train_scenes=[] --set val_scenes=[] "
                         f"--set split={name}; base with a split + a preset that brings lists (small_set): --set split=null")
    d = load(name, splits_dir)
    out.update(_split=name, train_scenes=d["train"], val_scenes=d["val"])
    return out


def list_hash(names):
    """12 hex digits of the set of names (the order of a list does not matter to the trainer)"""
    return hashlib.sha1("\n".join(sorted(names)).encode()).hexdigest()[:12]


def name_of(train, val, splits_dir=SPLITS):
    """the split whose lists are exactly these (as sets): its name; "official" for no lists (the whole official nuScenes split), else "inline" """
    if not train and not val:
        return "official"
    for f in sorted(os.listdir(splits_dir)) if os.path.isdir(splits_dir) else []:
        if f.endswith(".json"):
            d = load(f[:-5], splits_dir)
            if set(d["train"]) == set(train) and set(d["val"]) == set(val):
                return f[:-5]
    return "inline"


def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=1)
        f.write("\n")
    os.replace(path + ".tmp", path)


def _run_files(run_dir):
    """(config, gt) of a run folder: the config the trainer ran with and gt.json ({} when the run has none)"""
    cfg = _read(os.path.join(run_dir, "config.json"))
    if cfg is None:
        raise ValueError(f"{run_dir}: no readable config.json")
    return cfg, _read(os.path.join(run_dir, "gt.json")) or {}


def manifest(run_dir, card=None, splits_dir=SPLITS):
    """the minimal manifest of a run, from the files it wrote: config.json (the resolved config, so the lists), gt.json, code/SNAPSHOT.txt (commit and tag of the
    snapshot run_ar.sh made); `card` = the run's record (records/runs/<id>/run.json) adds hypothesis / prediction / rule, else null"""
    cfg, gt = _run_files(run_dir)
    train, val = cfg.get("train_scenes") or [], cfg.get("val_scenes") or []
    try:
        with open(os.path.join(run_dir, "code", "SNAPSHOT.txt"), encoding="utf-8") as f:
            m = re.match(r"commit (\w+).*?tag (\S+?)\)", f.read())
    except OSError:
        m = None
    rec = _read(card) if card else None
    return {"run": os.path.basename(os.path.normpath(run_dir)),
            "split": {"name": name_of(train, val, splits_dir), "train": {"n": len(train), "hash": list_hash(train)}, "val": {"n": len(val), "hash": list_hash(val)}},
            "gt": {"signature": gt.get("gt_signature"), "version": gt.get("gt_version")},
            "code": {"commit": m.group(1) if m else None, "tag": m.group(2) if m and m.group(2) != "none" else None},
            "card": {k: rec.get(k) for k in ("hypothesis", "prediction", "rule")} if rec else None}


def write_split_json(run_dir, out_dir, frames=None, splits_dir=SPLITS):
    """OUT_DIR/split.json: the scene lists the run trained and validated on (from its config.json), the split name, the GT version and the official nuScenes split names;
    `frames` = the keys of the evaluation frames when the caller has them (the evaluation card holds their count and hash only), else null. -> the path"""
    cfg, gt = _run_files(run_dir)
    train, val = cfg.get("train_scenes") or [], cfg.get("val_scenes") or []
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "split": name_of(train, val, splits_dir),
           "official_split": {"train": cfg.get("train_split"), "val": cfg.get("val_split")},
           "gt": {"signature": gt.get("gt_signature"), "version": gt.get("gt_version")},
           "train": list(train), "val": list(val), "evaluation_frames": None if frames is None else list(frames)}
    path = os.path.join(out_dir, "split.json")
    _write(path, out)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resolve", help="stdin: a config (json) -> stdout: the config with the lists of its split")
    p = sub.add_parser("names", help="the scene names of a split (train, then val), one per line: the format of gt_scenes.py names and autogt --scenes-file")
    p.add_argument("split")
    p = sub.add_parser("manifest", help="write RUN_DIR/manifest.json")
    p.add_argument("run_dir")
    p.add_argument("--card", help="the run's record (records/runs/<id>/run.json): adds hypothesis / prediction / rule")
    p = sub.add_parser("split-json", help="write OUT_DIR/split.json")
    p.add_argument("run_dir")
    p.add_argument("out_dir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "resolve":
            json.dump(resolve(json.load(sys.stdin)), sys.stdout)
        elif a.cmd == "names":
            d = load(a.split)
            print("\n".join(d["train"] + d["val"]))
        elif a.cmd == "manifest":
            path = os.path.join(a.run_dir, "manifest.json")
            _write(path, manifest(a.run_dir, a.card))
            print(f"written {path}")
        else:
            print(f"written {write_split_json(a.run_dir, a.out_dir)}")
    except (ValueError, OSError) as e:
        print(f"splits: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
