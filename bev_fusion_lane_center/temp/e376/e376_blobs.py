"""E376 (scratch): the README blobs for commit_paths.py --blob, built from the CURRENT HEAD + only my rows (other chats have uncommitted edits in these READMEs).
usage: python3 temp/e376/e376_blobs.py train|eval   -> temp/e376/blob_<name>.md files"""
import subprocess
import sys


def head(path):
    return subprocess.run(["git", "show", f"HEAD:bev_fusion_lane_center/{path}"], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def replace_rows(text, rows):
    lines = text.split("\n")
    for row in rows:
        key = row[: row.index("|", 2) + 1]                    # '| `name` |'
        hit = [k for k, l in enumerate(lines) if l.startswith(key)]
        assert len(hit) == 1, (key, len(hit))
        lines[hit[0]] = row
    return "\n".join(lines)


def rows_of(path, keys):
    out = []
    for line in open(path, encoding="utf-8").read().split("\n"):
        if any(line.startswith(k) for k in keys):
            out.append(line)
    assert len(out) == len(keys), (path, len(out))
    return out


what = sys.argv[1]
if what == "train":
    g = replace_rows(head("training/grid/README.md"), rows_of("temp/e376/readme_grid_e376.md", ["| `jtoken.py` |", "| `test_jtoken.py` |"]))
    t = replace_rows(head("training/README.md"), rows_of("temp/e376/readme_training_e376.md", ["| `train/jtoken_train.py` |"]))
    open("temp/e376/blob_training_grid_README.md", "w", encoding="utf-8").write(g)
    open("temp/e376/blob_training_README.md", "w", encoding="utf-8").write(t)
elif what == "eval":
    src = open("temp/e376/e376ro_readme_row.md", encoding="utf-8").read().split("\n")
    row = src[0]
    assert row.startswith("| `jtoken_eval.py` |")
    old = "the parity and timing code paths)"
    new = src[3]
    assert new.startswith("the parity and timing code paths;")
    e = replace_rows(head("evaluation/README.md"), [row])
    assert e.count(old) == 1, e.count(old)
    e = e.replace(old, new)
    open("temp/e376/blob_evaluation_README.md", "w", encoding="utf-8").write(e)
print("ok", what)
