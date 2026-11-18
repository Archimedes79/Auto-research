import io
p = "experiments/E360_matrix-nms-in-the-e2e-stack/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
add = io.open("temp/_session_scratch_s292/e360_summary.md", encoding="utf-8").read().replace("\n", nl)
a = "## 6. Summary" + nl + "(empty until the runs exist)"
assert s.count(a) == 1
s = s.replace(a, "## 6. Summary" + nl + add.rstrip() + nl)
b = "## 5. Build log" + nl + "(empty)"
assert s.count(b) == 1
s = s.replace(b, "## 5. Build log" + nl + "No code was built (the scorer is unchanged; `temp/e360/verdict_e360.py` and `temp/e360/start_chain.sh` are scratch, kept by the run tags). Chain started 09:19 (accidentally: the dry-run flag did not pass through `wsl`), `E360.t_s0`, `c_s0`, `a_s0` ran through; `E360.ck_s0` failed at 09:5x with OSError 12 (WSL page cache full, `gzip` read of the dump; `run.py` does not retry it), the page cache was dropped as root, a cache guard (`temp/_session_scratch_s292/cache_guard_day.sh`, until 17:30) started, and the failed / blocked ids were resumed by `temp/e360/restart_chain.sh` (the first attempt's log is kept as `run.1.log`; the cards of `ck_s0` and later were written by the second attempt only).")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
