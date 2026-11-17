import os
import subprocess

R = "/mnt/c/Auto-research"
P = "bev_fusion_lane_center/"
OUT = R + "/bev_fusion_lane_center/temp/_session_scratch_s292/blobs_e353"
os.makedirs(OUT, exist_ok=True)


def head(path):
    return subprocess.run(["git", "-C", R, "show", "HEAD:" + path], capture_output=True, text=True, check=True).stdout


def work(path):
    return open(os.path.join(R, path), encoding="utf-8", newline="").read()


# 1. training/README.md: HEAD with line 19 (the train/ row) taken from the working tree: the working row holds ONLY my insertion (checked: `jrecur` is not in it)
h, w = head(P + "training/README.md").split("\n"), work(P + "training/README.md").split("\n")
assert len(h) == len(w)
row = w[18]
k = row.index("; `jrecur_train.py`")                    # the E354 sentence of the other session: cut
end = row.rindex(" |")
row = row[:k] + row[end:]
assert "jrecur" not in row and "jnarrow_train.py" in row and "jnarrow_train" not in h[18]
h[18] = row
open(OUT + "/training_README.md", "w", encoding="utf-8", newline="").write("\n".join(h))

# 2. training/grid/README.md: HEAD + my two rows
h = head(P + "training/grid/README.md").split("\n")
w = work(P + "training/grid/README.md").split("\n")
mine = [l for l in w if l.startswith("| `jnarrow.py`") or l.startswith("| `test_jnarrow.py`")]
assert len(mine) == 2 and not any(l.startswith("| `jnarrow.py`") for l in h)
i = max(k for k, l in enumerate(w) if l.startswith("| `test_jnarrow.py`"))
anchor = w[i - 2]                      # the row before my two rows in the working file
j = h.index(anchor)
h[j + 1:j + 1] = mine
open(OUT + "/grid_README.md", "w", encoding="utf-8", newline="").write("\n".join(h))

# 3. tools/maintenance/run_all_tests.sh: HEAD + my line after the test_jprune_probe.py line
h = head(P + "tools/maintenance/run_all_tests.sh").split("\n")
w = work(P + "tools/maintenance/run_all_tests.sh").split("\n")
mine = [l for l in w if "E353 narrowing" in l]
assert len(mine) == 1
j = max(k for k, l in enumerate(h) if l.rstrip().endswith("test_jprune_probe.py"))
h[j + 1:j + 1] = mine
open(OUT + "/run_all_tests.sh", "w", encoding="utf-8", newline="").write("\n".join(h))
print("ok", os.listdir(OUT))
