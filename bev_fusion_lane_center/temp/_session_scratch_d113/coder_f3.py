p = "evaluation/jpast_eval.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)
rep('''    wsl bash tools/env/_py14.sh jpast_eval.py dump --run R --seed S [--split val|train] [--mode stream|truncated|off|shuffled|still] [--window 3] [--cpu] [--max N]   # GPU: -> runs/R/token_<split>[_<mode>][_n<N>].pkl
    wsl bash tools/env/_py14.sh jpast_eval.py use  --run R [--split val]                                                                                          # CPU: u per stream position and what the past changed (stream against off) -> runs/R/past_use_<split>.json
''', '''    wsl bash tools/env/_py14.sh jpast_eval.py dump --run R --seed S [--split val|train] [--mode stream|truncated|off|shuffled|still] [--window 3] [--cpu] [--max N]   # GPU: -> runs/<R or R_<mode>>/token_<split>[_n<N>].pkl
    wsl bash tools/env/_py14.sh jpast_eval.py use  --run R [--split val] [--max N]                                                                                # CPU: u and the carried state's RMS per stream position, what the past changed (stream against off) -> runs/R/past_use_<split>.json
    wsl bash tools/env/_py14.sh jtoken_eval.py score --run R_off --seed S   |   sweep --run R_off --seed S --split train ...                                       # a mode dump is read by the scorer like any run: its folder is the sibling R_<mode>
''')
rep('''  stream     the state of the previous frame is carried through the whole scene (the verdict mode; its file is `token_<split>.pkl`, the one `jtoken_eval.py score` / `sweep` read)''', '''  stream     the state of the previous frame is carried through the whole scene (the verdict mode; its folder is the run's own: `runs/R/token_<split>.pkl`, the file `jtoken_eval.py score` / `sweep` read)''')
rep('''TRAIN (873 evenly spaced frames, as `jtoken_eval.sample`): the stream runs through the whole split, only the sampled frames are dumped.''', '''Every other mode goes to the SIBLING run folder `runs/R_<mode>/` (stream: `runs/R/`): `token_<split>[_n<N>].pkl` with meta['run'] = `R_<mode>`, and `model.pt` = a copy of R's (an existing one must be the same bytes), so that `jtoken_eval.py score --run R_<mode>`
and `sweep` read it as they read a run (they need model.pt for its args and the dump named as above; their cards land in the sibling folder; the parameter count they print is that of the E365 head without the block).
TRAIN (873 evenly spaced frames, as `jtoken_eval.sample`): the stream runs through the whole split, only the sampled frames are dumped.''')
rep('''sums d2 / x2 / n of the kill measure u = RMS(block(x, past) - block(x, none key only)) / RMS(x) (NaN-free: 0 without a past).''', '''sums d2 / x2 / n of the kill measure u = RMS(block(x, past) - block(x, none key only)) / RMS(x) (0 without a past) and state_ss / state_n = sum of squares and count of the CARRIED state (the head's final token states x of that frame, valid tokens).
`use` writes u and the RMS of the carried state per stream position (0 = the cold start, 1, 2, ... frames since the last reset), the stream against off (what the past changed) and the frame indices of the first 5 frames of every scene against the later ones (the card's topo-F1 split).''')
rep('''def mode_sfx(mode):
    return "" if mode == "stream" else f"_{mode}"
''', '''def run_folder(run, mode):
    """the run folder a dump of `mode` is written to: the run itself for stream, the sibling `<run>_<mode>` for every other mode"""
    return run if mode == "stream" else f"{run}_{mode}"


def sibling(runs, run, mode):
    """the folder of `run_folder` with the run's model.pt in it (a copy; an existing one must be the same bytes: another checkpoint under the same name is refused) -> its name"""
    name = run_folder(run, mode)
    if name != run:
        os.makedirs(os.path.join(runs, name), exist_ok=True)
        src, dst = os.path.join(runs, run, "model.pt"), os.path.join(runs, name, "model.pt")
        if os.path.exists(dst):
            assert filecmp.cmp(src, dst, shallow=False), f"{dst} is not a copy of {src}"
        else:
            shutil.copyfile(src, dst)
    return name


def dump_file(runs, name, split, n=0):
    """runs/<name>/token_<split>[_n<n>].pkl (n: the frames of a `--max` dump), the name `jtoken_eval.dump_path` builds for model.pt, no drop, every 1"""
    return os.path.join(runs, name, f"token_{split}{f'_n{n}' if n else ''}.pkl")
''')
rep("import argparse\nimport json", "import argparse\nimport filecmp\nimport json")
rep("import pickle\nimport sys", "import pickle\nimport shutil\nimport sys")
rep('''    (cache index, frame batch on the CPU, the head's output, dict(has_prev, pos = position in the scene, shift = the ego shift in m, d2 / x2 / n = the sums of the kill measure)).''', '''    (cache index, frame batch on the CPU, the head's output, dict(has_prev, pos = position in the scene, shift = the ego shift in m, d2 / x2 / n = the sums of the kill measure, state_ss / state_n = the sum of squares and the count of the final
    token states x of the frame's valid tokens: the state that is carried on)).''')
rep('''            yield i, fr, out, dict(has_prev=hp, pos=pos_in_scene, shift=float(b["rel_t"][0, 0].norm()), d2=d2, x2=x2, n=n)''', '''            v = out["valid"][0].bool()
            yield i, fr, out, dict(has_prev=hp, pos=pos_in_scene, shift=float(b["rel_t"][0, 0].norm()), d2=d2, x2=x2, n=n, state_ss=float((out["x"][0].float() ** 2)[v].sum()), state_n=int(v.sum()) * int(out["x"].shape[-1]))''')
rep('''use = max(keep), [], dict(idx=[], has_prev=[], pos=[], shift=[], d2=[], x2=[], n=[])''', '''use = max(keep), [], dict(idx=[], has_prev=[], pos=[], shift=[], d2=[], x2=[], n=[], state_ss=[], state_n=[])''') if False else None
open(p, "w", encoding="utf-8").write(s)
print("ok")
