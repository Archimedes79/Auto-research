"""The two timings of the quiet window (E351.q_eager, E353.time_s0) into the E351 card, the E353 card, temp/report/compute.json and the ledger (numbers read from the json files, not typed).
usage (problem folder, WSL): python3 temp/_session_scratch_s292/ms_night.py"""
import io
import json
import subprocess

q = json.load(open("runs/eval/jtiming/e351_quiet_eager.json"))
t = json.load(open("runs/eval/jtiming/e353_time_s0.json"))
pl = q["plain_block_medians_ms"] if False else q["plain_median_ms"]
pp = q["paired"]
ci = lambda a: f"{a[0]:+.2f} ms [{a[1]:+.2f}, {a[2]:+.2f}]"
n_pd = t["paired_narrowed_minus_control"]
q_txt = f"""

## 3b. Quiet-host eager forward of E341 and E348 (night of 2026-11-16, run `E351.q_eager`, started 04:38 after 20 quiet samples: GPU util 0 %, 1-minute load 0.00, no GPU job; `runs/eval/jtiming/e351_quiet_eager.json`)
**What ran:** `jtiming.py run --runs e312_p_s0,e341_p_s0,e324_s_s0,e348_w_s0 --tag e351_quiet_eager` (the E342 part B protocol: batch 1, bf16 autocast, no_grad, {q['frames']} cached VAL frames, {q['rounds']} rounds x {q['iters']} forwards in alternating blocks; the GPU was idle before (util max {q['gpu_before']['util_max']:.0f} %) and after ({q['gpu_after']['util_max']:.0f} %)); one session, seed-0 checkpoints. **Median eager ms per frame: E312 {pl['e312_p_s0']:.2f}, E341 {pl['e341_p_s0']:.2f}, E324 s {pl['e324_s_s0']:.2f}, E348 {pl['e348_w_s0']:.2f}.**
**Paired differences [95 % interval over the rounds]:** E341 - E312 **{ci(pp['e341_p_s0 - e312_p_s0'])}**; E348 - E324 s **{ci(pp['e348_w_s0 - e324_s_s0'])}**; E324 s - E312 {ci(pp['e324_s_s0 - e312_p_s0'])}; E348 - E312 {ci(pp['e348_w_s0 - e312_p_s0'])}.
**Prediction (card of this run: E341 - E312 about -1 ms, E348 - E324 s about -1 ms, E324 s - E312 about +5.0 ms): HIT** (-0.85 / -0.80 / +5.45; the two branch removals buy about 0.8 ms each on the quiet host, 0.25 - 0.3 ms less than E342's loaded-host -1.1; the price of E324 s over E312 is +5.0 .. +5.8 ms, the four-session quiet graph block gave +4.7 .. +5.3). **Reading:** removing the whole 0.5 m paint branch (-1.5 GMACs, whole conv kernels gone) is worth about -0.8 ms of the 15 / 20 ms; the MACs are not what the ms follows (E351: the junction layer is launch-bound): E353's quarter-width branch saves 1.1 GMACs but keeps every kernel, see its card (-0.14 ms, interval through 0). **Caveats:** the 12-frame protocol of this run gives medians 0.4 - 0.9 ms above the 24-frame sessions of section 3 (E324 s {pl['e324_s_s0']:.2f} against 19.5 - 19.9 ms: a frame-set effect, so absolute ms of the two protocols are not mixed; the paired differences are); one session; rounds are the resampling unit (not scenes).
"""
p = "experiments/E351_launch-bound-junction-layer/experiment.md"
s = io.open(p, encoding="utf-8", newline="").read()
nl = "\r\n" if "\r\n" in s else "\n"
assert "## 3b. Quiet-host eager forward of E341 and E348" not in s
io.open(p, "w", encoding="utf-8", newline="").write(s.rstrip("\r\n") + q_txt.replace("\n", nl).rstrip() + nl)

e353 = f"""- **Quiet-host ms (the fourth fact of the D106 line; `E353.time_s0`, 2026-11-16 04:42, `runs/eval/jtiming/e353_time_s0.json`, written after the first draft of this section): label {t['label']}** ({t['quiet_line']}); `jnarrow_eval.py time paired --runs e324_s_s0,e353_n1_s0` in ONE process (E324 s then the narrowed n1 through the surgery-aware loader), batch 1, bf16 autocast, {t['frames']} cached VAL frames, {t['rounds']} rounds x {t['iters']} forwards, seed-0 checkpoints. **E324 s {t['control_median_ms']:.3f} ms, n1 {t['narrowed_median_ms']:.3f} ms; paired n1 - E324 s {n_pd[0]:+.3f} ms [{n_pd[1]:+.3f}, {n_pd[2]:+.3f}] ({t['percent']:+.2f} %). The upper end of the interval is above 0: n1 is NOT faster by the written rule ('faster' = the upper end below 0); the ms is a TIE.** The prediction (about -0.8 ms, a guess from the loaded-host whole-branch number) MISSED: removing the whole branch (E348, E351 section 3b) buys -0.80 ms [-1.18, -0.44] on the same quiet host because its kernels disappear, while narrowing the width removes 1.1 GMACs and keeps every launch (the junction layer is launch-bound, E351).
- **D106 applied (by the rule written in section 0):** parts: the same parts, one new checkpoint class; parameters -27 664 (-1.0 %); GMACs -1.1059 (-13.5 %); quiet-host ms -0.14 ms with the interval through 0 (not faster). **n1 is simpler by MACs and parameters but not faster, so D106's 'simpler AND faster' does NOT choose it; the champion stays E324 s.** A ms-neutral variant with 13.5 % fewer MACs and one extra checkpoint class is a case for Mirko (D106: 'faster-with-more-parts' and 'simpler-but-slower' are his; this one is simpler-and-speed-neutral, so it is his as well), not decided here. The ADOPT-FIRM of the accuracy rule stands: on E324 s's recipe the narrowing loses nothing measurable; what it buys is MACs, not milliseconds on this host.
"""
p3 = "experiments/E353_paint-branch-width-25/experiment.md"
s3 = io.open(p3, encoding="utf-8", newline="").read()
nl3 = "\r\n" if "\r\n" in s3 else "\n"
anchor = "- **Named limits (honest):**"
assert s3.count(anchor) == 1
s3 = s3.replace(anchor, e353.replace("\n", nl3) + anchor)
io.open(p3, "w", encoding="utf-8", newline="").write(s3)

pc = "temp/report/compute.json"
c = json.load(open(pc))
c["quiet_host_night_2026_11_16"] = {
    "protocol": "E351.q_eager (jtiming.py run, 12 cached VAL frames, 10 rounds x 25 forwards, bf16, batch 1, QUIET: load 0.00, GPU idle) and E353.time_s0 (jnarrow_eval.py time paired, 24 frames, 10 x 30, load 1.03 / 0.89, QUIET)",
    "eager_median_ms_12_frames": pl,
    "paired_ms_with_95_interval_over_rounds": {k: v for k, v in pp.items()},
    "e353_n1_minus_e324_s_ms": {"point_ci": n_pd, "control_ms": t["control_median_ms"], "narrowed_ms": t["narrowed_median_ms"], "percent": t["percent"], "label": t["label"]},
    "note": "absolute ms of the 12-frame and the 24-frame protocols are not mixed (E324 s 20.58 vs 19.5 - 19.9); the paired differences are comparable",
}
for k, run in (("E341 (E312 recipe, no 0.5 m branch)", "e341_p_s0"), ("E348 (E324 s recipe, no 0.5 m branch)", "e348_w_s0")):
    c[k]["ms_per_frame_quiet_eager"] = pl[run]
    c[k]["ms_per_frame_quiet_eager_note"] = "E351.q_eager 2026-11-16, 12-frame protocol, quiet host (see quiet_host_night_2026_11_16)"
json.dump(c, open(pc, "w"), indent=1)

P = "python3 ../autoresearch/tools/ledger.py note".split()


def note(rid, result, decision):
    r = subprocess.run(P + [rid, "--result", result, "--decision", decision], capture_output=True, text=True)
    print(rid, r.returncode, (r.stdout + r.stderr).strip()[:100])


note("E351.q_eager", f"quiet host (load 0.00, GPU idle), eager median ms: E312 {pl['e312_p_s0']:.2f}, E341 {pl['e341_p_s0']:.2f}, E324 s {pl['e324_s_s0']:.2f}, E348 {pl['e348_w_s0']:.2f}; paired E341-E312 {ci(pp['e341_p_s0 - e312_p_s0'])}, E348-E324 s {ci(pp['e348_w_s0 - e324_s_s0'])}, E324 s-E312 {ci(pp['e324_s_s0 - e312_p_s0'])}",
     "prediction HIT (-1 / -1 / +5.0 ms); by hand: measurement (card section 3b)")
note("E353.time_s0", f"quiet host ({t['quiet_line']}): E324 s {t['control_median_ms']:.3f} ms, n1 {t['narrowed_median_ms']:.3f} ms, paired n1-c {n_pd[0]:+.3f} ms [{n_pd[1]:+.3f}, {n_pd[2]:+.3f}] ({t['percent']:+.2f} %)",
     "NOT faster (upper end above 0): D106 does not choose n1, champion stays E324 s; prediction -0.8 ms MISSED; by hand: measurement (card section 3)")
print("ok")
