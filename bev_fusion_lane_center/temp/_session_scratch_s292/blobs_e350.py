import io, subprocess
def head(p):
    return subprocess.run(["git", "show", "HEAD:bev_fusion_lane_center/" + p], capture_output=True).stdout.decode("utf-8")
nl = "\n"
rows = [l for l in io.open("temp/e350/readme_rows.md", encoding="utf-8").read().split(nl) if l.startswith("| `")]
r_data = [l for l in rows if l.startswith("| `jwire_data.py` |")]
r_check = [l for l in rows if l.startswith("| `jwire_check.py` |")]
r_verd = [l for l in rows if l.startswith("| `jwire_verdict.py` |")]
r_train = [l for l in rows if l.startswith("| `train/jwire_train.py` |")]
assert len(r_data) == len(r_check) == len(r_verd) == len(r_train) == 1
def edit(path, out, replace, add_after=None):
    lines = head(path).split(nl)
    i = [k for k, l in enumerate(lines) if l.startswith(replace[0])]
    assert len(i) == 1, (path, replace[0], i)
    lines[i[0]] = replace[1]
    if add_after:
        lines[i[0] + 1:i[0] + 1] = add_after
    io.open(out, "w", encoding="utf-8", newline="").write(nl.join(lines))
edit("evaluation/README.md", "temp/_session_scratch_s292/readme_e350.md", ("| `jwire_data.py` |", r_data[0]), r_check + r_verd)
edit("training/README.md", "temp/_session_scratch_s292/treadme_e350.md", ("| `train/jwire_train.py` |", r_train[0]))
# run_gates.sh: two lines after the jwire_train selftest line
lines = head("tools/maintenance/run_gates.sh").split(nl)
i = [k for k, l in enumerate(lines) if "jwire_train.py selftest" in l]
assert len(i) == 1
new = ['run "head: wiring head checks: twin card and live records, planted differences (E350)"  jwire_check.py selftest',
       'run "head: wiring head verdict: sign classes, kill, shift flag, planted cards (E350)"  jwire_verdict.py selftest']
lines[i[0] + 1:i[0] + 1] = new
io.open("temp/_session_scratch_s292/gates_e350.sh", "w", encoding="utf-8", newline="").write(nl.join(lines))
print("ok")
