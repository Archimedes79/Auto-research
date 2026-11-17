set -e
cd /mnt/c/Auto-research/bev_fusion_lane_center
python3 - <<'EOF'
p = "training/data/lane_past_data.py"
s = open(p, encoding="utf-8").read()
def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:70])
    s = s.replace(a, b)
rep('''def load_ext(d):''', '''def check_built(cache=LC.CACHE, log=print):
    """every finished cache directory has its extension, current (the sha of frames.json), with the frames and scene starts of the data and, for VAL, exactly the rows of data/splits.json['val'] -> the number of directories checked"""
    dirs = cache_dirs(cache)
    assert len(dirs) >= 1, f"no finished cache directory under {cache}"
    for d in dirs:
        ext, split = load_ext(d), os.path.basename(d)
        rows = json.load(open(os.path.join(d, "frames.json")))["rows"]
        check_rows(split, rows)
        assert (len(ext["prev"]), int((ext["prev"] < 0).sum())) == EXPECT[split] and ext["info"]["with_past"] == EXPECT[split][0] - EXPECT[split][1], f"{d}: {ext['info']}"
        log(f"ok {os.path.relpath(d, ROOT)}: {len(ext['prev'])} frames, {int((ext['prev'] < 0).sum())} scene starts")
    return len(dirs)


def load_ext(d):''')
rep('''    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()''', '''    sub.add_parser("check")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "check":
        check_built()''')
rep('''    wsl bash tools/env/_py14.sh lane_past_data.py fixture --split val''', '''    wsl bash tools/env/_py14.sh lane_past_data.py check                                                        # every built directory: extension current, frames / scene starts of the data, VAL = the rows of data/splits.json
    wsl bash tools/env/_py14.sh lane_past_data.py fixture --split val''')
open(p, "w", encoding="utf-8").write(s)
print("ok")
EOF
