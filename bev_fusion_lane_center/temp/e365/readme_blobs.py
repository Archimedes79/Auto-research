"""blobs of the shared README files = HEAD + only MY changed rows (the lane chat's E368 rows stay out: their files are not committed yet)"""
import subprocess

MINE = {"training/README.md": ["train/jtoken_train.py"],
        "training/grid/README.md": ["jtoken.py", "test_jtoken.py"],
        "evaluation/README.md": ["jtoken_eval.py", "tests/"]}
for path, keys in MINE.items():
    head = subprocess.run(["git", "show", f"HEAD:bev_fusion_lane_center/{path}"], capture_output=True, text=True, encoding="utf-8").stdout.split("\n")
    cur = open(path, encoding="utf-8").read().split("\n")
    for k in keys:
        tag = f"| `{k}` |"
        hi = [i for i, l in enumerate(head) if l.startswith(tag)]
        ci = [i for i, l in enumerate(cur) if l.startswith(tag)]
        assert len(hi) == 1 and len(ci) == 1, (path, k, hi, ci)
        if "jpast" in cur[ci[0]] and "jpast" not in head[hi[0]]:
            print("WARNING: my row mentions jpast", path, k)
        head[hi[0]] = cur[ci[0]]
    out = "temp/e365/blob_" + path.replace("/", "_")
    open(out, "w", encoding="utf-8").write("\n".join(head))
    print(path, "->", out)
