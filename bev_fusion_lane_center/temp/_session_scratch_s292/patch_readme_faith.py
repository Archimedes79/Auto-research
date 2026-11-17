import io
p = "evaluation/README.md"
lines = io.open(p, encoding="utf-8").read().split("\n")
def add(prefix, text):
    idx = [i for i, l in enumerate(lines) if l.startswith(prefix)]
    assert len(idx) == 1, (prefix, idx)
    l = lines[idx[0]].rstrip()
    assert l.endswith("|")
    lines[idx[0]] = l[:-1].rstrip() + " " + text + " |"
add("| `jwire_data.py` |", "E358 attempt 4: `live_records(..., faithful=True)` replays HOW the stored dumps were made (B on a loader of its dump's settings `DUMP_B`, the pair model on the pair dump's `DUMP_PAIR`, joined by row, at most 64 val_clean frames; a check, not a TRAIN pass).")
add("| `jwire_check.py` |", "`live` also takes `--bs` (batch size of the live pass, default 4) and `--faithful` (the two-loader replay of the stored dumps' settings; excludes `--bs`) and prints the size of the differences of a REFUSED frame as a diagnostic only.")
io.open(p, "w", encoding="utf-8").write("\n".join(lines))
print("ok")
