"""experiments/E2E_cost_table.md from the timing jsons (scratch, 2026-11-18): the numbers are READ from the files, none typed. usage: BFLC_ROOT=. python make_cost_table.py"""
import glob
import json
import os

R = os.environ.get("BFLC_ROOT", ".")


def last(pat):
    """-> ((file, json) of the last QUIET run, (file, json) of the first run: the one without the aging wait)"""
    ds = [(f, json.load(open(f))) for f in sorted(glob.glob(os.path.join(R, pat)))]
    q = [x for x in ds if x[1].get("quiet")]
    return (q[-1] if q else None), ds[0]


(e2f, e2d), (e2f0, e2d0) = last("runs/e373_cb_s0/token_timing_ens_2*.json")
(s2f, s2d), (s2f0, s2d0) = last("runs/e368_r_s0/past_timing_2*.json")
(smf, smd), (smf0, smd0) = last("runs/e374_m_s0/map_timing_2*.json")
(szf, szd), _ = last("runs/e374_z_s0/map_timing_2*.json")
for name, d in (("e2", e2d), ("s2", s2d), ("s3m", smd), ("s3z", szd)):
    assert d["quiet"] and d["usable_for_gcost"], name
m, p = e2d["median_ms"], e2d["p90_ms"]
m0 = e2d0["median_ms"]
a_r, a_z = s2d["arms"]["e368_r_s0"], s2d["arms"]["e368_z_s0"]


def f1(x):
    return f"{x:.2f}"


def rel(f):
    return f.split("runs/", 1)[1] if "runs/" in f else f


def host(d):
    return d["host"].replace("(idle: at most 5 %)", "").strip()


s3_cost = smd["median_ms"] - szd["median_ms"]
e2_seq_cost = m["whole_seq2"] - m["whole_1"]
e2_stk_cost = m["whole_stacked"] - m["whole_1"]
out = f"""# E2E cost table - what the pieces of the token champion cost at inference (measured 2026-11-18 05:18 - 05:30)

Measured numbers only; "not measured" where it is. RTX 3060, batch 1, bf16 autocast, CUDA events with a sync in front and behind, no_grad, cached VAL frames, 24 frames x 5 repeats (S2: 24 x 4, S3: 20 x 5) after one untimed pass, the median of the timed frames; the host must read QUIET (load below 3.0 before and after, GPU idle at most 5 % before and after, `jtiming.quiet_verdict`). Every row below carries that verdict. Written by the lane chat for the lead's request (S2 / S3 / ensemble cost) and the report.

## The numbers
| piece (what is timed) | median ms | p90 ms | parameters | against |
|---|---|---|---|---|
| **champion path**: B' (QGraph `e292_a_s0`) + ONE token head (`e373_cb_s0`), whole path | **{f1(m['whole_1'])}** | {f1(p['whole_1'])} | {e2d['params_whole_path']['one_head']:,} (B' {e2d['params_bprime']:,} + head {e2d['params']['head']:,}) | - |
| B' alone | {f1(m['bprime'])} | {f1(p['bprime'])} | {e2d['params_bprime']:,} | - |
| one token head alone (probabilities) | {f1(m['head_probs'])} | {f1(p['head_probs'])} | {e2d['params']['head']:,} | - |
| **e2**: two heads in sequence (`e373_cb_s0` + `e367_b25_s0`), whole path | **{f1(m['whole_seq2'])}** | {f1(p['whole_seq2'])} | {e2d['params_whole_path']['two_heads']:,} | champion +{f1(e2_seq_cost)} ms ({100 * (m['whole_seq2'] / m['whole_1'] - 1):.0f} %) |
| e2: two heads STACKED (`torch.func.stack_module_state` + `vmap`), whole path | {f1(m['whole_stacked'])} | {f1(p['whole_stacked'])} | {e2d['params_whole_path']['two_heads']:,} | champion +{f1(e2_stk_cost)} ms; slower than the sequence (the head is latency-bound at batch 1); output equals the two single forwards to {e2d['stacked_max_diff']:.1e} |
| e2 heads only: sequence / stacked (probabilities) | {f1(m['seq2_probs'])} / {f1(m['stacked_probs'])} | {f1(p['seq2_probs'])} / {f1(p['stacked_probs'])} | {e2d['params']['two_heads']:,} | one head {f1(m['head_probs'])} |
| **S2** (E368): the head WITH the past block `r` (`e368_r_s0`), head alone, stream order | **{f1(a_r['median_ms'])}** | {f1(a_r['p90_ms'])} | {a_r['params']:,} (block {a_r['params_block']:,}) | block cost **+{f1(s2d['block_cost_ms'])} ms** over `z` ({f1(a_z['median_ms'])} ms, {a_z['params']:,}); the first frame of a scene (no past) {f1(a_r['cold_median_ms'])} ms |
| **S3** (E374): the head WITH the SD-map block `m` (`e374_m_s0`), head alone | **{f1(smd['median_ms'])}** | {f1(smd['p90_ms'])} | {smd['params']:,} (block {smd['params_block']:,}) | block cost **+{f1(s3_cost)} ms** over `z` ({f1(szd['median_ms'])} ms, {szd['params']:,}); about {smd['map_samples_mean']:.0f} map samples per frame |
| **S4** (E372): B' node head trained under the head's loss | not measured here | - | - | expected to cost what the champion path costs (the same modules, B' + head, trained jointly: other weights, not other layers); NOT timed, its timing is not part of tonight's chain |

## Reading
- **e2 does not meet the cost bar of the E379 gate (D106): the stacked whole path is {f1(m['whole_stacked'])} ms and even the sequence is {f1(m['whole_seq2'])} ms, both above the 19.7 ms the card names (E324 s, quoted from the card, not re-measured tonight).** The ensemble buys topo F1 +0.0113 / +0.0065 (card section 3) at +{f1(e2_seq_cost)} ms and +{e2d['params']['head']:,} parameters; the lead decides (D106).
- S2 and S3 cost +{f1(s2d['block_cost_ms'])} / +{f1(s3_cost)} ms on the HEAD (the S2 card predicted +1 - 2 ms: matches) and neither block was adopted (E368 and E374 TIE by their written rules).
- The champion path is {f1(m['whole_1'])} ms here ({100 * m['bprime'] / m['whole_1']:.0f} % of it is B' alone, timed separately); the head is the cheaper part.

## How the numbers were made, and the first (not-QUIET) runs
Run ids `E379.time_e2b` (`token_time_ens.py`, the card's registered rule), `diag.s2_timing_b` (`temp/e368/time_s2.py`, scratch), `diag.s3_time_m_b` / `diag.s3_time_z_b` (`jmap_eval.py time` through `temp/e374/time_s3_aged.py`, scratch); files `{rel(e2f)}`, `{rel(s2f)}`, `{rel(smf)}`, `{rel(szf)}`.
Hosts: e2 {host(e2d)}; S2 {host(s2d)}; S3 m {host(smd)}; S3 z {host(szd)}.
**The first runs of the same night (05:09 - 05:14: `E379.time_e2`, `diag.s2_timing`, `diag.s3_time_m`) were flagged NOT QUIET only because the tools took the GPU after-sample at once: nvidia-smi averages its utilisation over a window, so the sample read the process's own last kernels (65 - 70 %) on an idle box; `jtiming.py`'s own protocol waits 2.5 s first (`time.sleep(2.5)`), the timing commands of the other tools did not.** The repeat with the wait (`token_time_ens.after_state`, the two scratch scripts) reads QUIET, and the milliseconds agree with the first runs: e2 whole path {f1(m0['whole_1'])} / {f1(m0['whole_seq2'])} / {f1(m0['whole_stacked'])} ms (first run: one head / sequence / stacked) against {f1(m['whole_1'])} / {f1(m['whole_seq2'])} / {f1(m['whole_stacked'])} (repeat), S2 block cost +{f1(s2d0['block_cost_ms'])} against +{f1(s2d['block_cost_ms'])} ms, S3 m {f1(smd0['median_ms'])} against {f1(smd['median_ms'])} ms. `jtoken_eval.py timing` has the same missing wait until the unit of the token-head owner lands: a NOT QUIET flag right after its timed passes may be this artefact, the milliseconds are not affected.
Limits: one cache seed (0), one pair of heads for e2, 24 / 20 frames: the p90 is wider than the median (a stall in a few frames); two quiet repeats differ by about 0.1 - 0.4 ms (the two e2 runs by {f1(abs(m['whole_seq2'] - m0['whole_seq2']))} ms on the sequence). S4's own cost and the timing of the K 384 variants (E383) are not in this table.
"""
with open(os.path.join(R, "experiments", "E2E_cost_table.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(out)
print(out[:1800])
