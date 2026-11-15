"""The ground truth a config needs: its scene names, the check and the build - the `prepare` step of `python -m trainsuite start`.

    python training/gt_scenes.py CONFIG names     # the sequence names the config trains and validates on, one per line
    python training/gt_scenes.py CONFIG check     # exit 0 when the GtGenerator store holds every keyframe of them for the CURRENT generator, else 1 and how many are missing
    python training/gt_scenes.py CONFIG build     # generate what is missing (the front GT only: --no-lane): resumable, nice 15, light devkit (hours for the whole trainval: a run of its own)

GtGenerator does not read training configs (it must not): this wrapper turns the config into a file of sequence names (`<config>.scenes.txt`, next to the config, one name per
line) and calls `python -m GtGenerator have|generate --scenes-file` from the repository top, as GtGenerator's guide asks. Scenes: the official split of the config (`train_split`,
`val_split`; mini versions use mini_train / mini_val), restricted by `train_scenes` / `val_scenes`, the two parts together, without repeats. The trainer additionally leaves
out scenes whose CAM_FRONT files are not on disk; the store check wants all of them (`python -m GtGenerator data` says which raw files are missing).
Which store: the GtGenerator store is the single source going forward; until the dataset reads it (D29 part 5), the trainer builds its own cache (`gt_cache`) from the live
generator, so a passing check says that the GtGenerator store holds the GT, not that this cache is filled.
Run from the bev_fusion folder in the WSL with the gcar environment; tests/test_gt_prepare.py.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                                   # bev_fusion/
REPO = os.path.dirname(ROOT)                                   # the repository top: GtGenerator runs from there
sys.path.insert(0, HERE)


def scene_names(cfg):
    """the sequence names a config trains and validates on (train part first), without repeats"""
    from nuscenes.utils.splits import create_splits_scenes
    sp = create_splits_scenes()
    out = []
    for split, only in ((cfg.train_split, cfg.train_scenes), (cfg.val_split, cfg.val_scenes)):
        if cfg.version.endswith("mini") and split not in ("mini_train", "mini_val"):
            split = "mini_" + split
        for n in sp[split]:
            if (not only or n in set(only)) and n not in out:
                out.append(n)
    return out


def names_path(config_path):
    return os.path.splitext(os.path.abspath(config_path))[0] + ".scenes.txt"


def write_names(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    names = scene_names(cfg)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(n + "\n" for n in names))
    return names


def have_argv(cfg, names_file, py=None):
    return [py or sys.executable, "-m", "GtGenerator", "have", "--scenes-file", names_file, "--version", cfg.version, "--dataroot", os.path.expanduser(cfg.data_root)]


def build_argv(cfg, names_file, workers=2):
    return ["bash", "GtGenerator/run_nice.sh", "-m", "GtGenerator", "generate", os.path.expanduser(cfg.data_root), "--version", cfg.version, "--scenes-file", names_file,
            "--workers", str(workers), "--no-lane"]                       # the front GT only: the lane stores belong to the lane GT (a BEV prepare must not write them)


def main(argv=None, runner=subprocess.call):
    argv = list(sys.argv[1:] if argv is None else argv)
    workers = 2
    if "--workers" in argv:
        i = argv.index("--workers")
        workers = int(argv[i + 1])
        del argv[i:i + 2]
    if len(argv) != 2 or argv[1] not in ("names", "check", "build"):
        print(__doc__)
        return 2
    from train import read_config
    cfg = read_config(argv[0])
    path = names_path(argv[0])
    names = write_names(cfg, path)
    if argv[1] == "names":
        print("".join(n + "\n" for n in names), end="")
        return 0
    print(f"{len(names)} sequences of {cfg.version} ({os.path.relpath(path, ROOT) if path.startswith(ROOT) else path})", flush=True)
    cmd = have_argv(cfg, path) if argv[1] == "check" else build_argv(cfg, path, workers)
    return runner(cmd, cwd=REPO)


if __name__ == "__main__":
    sys.exit(main())
