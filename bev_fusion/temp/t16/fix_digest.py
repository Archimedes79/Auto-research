"""scratch: corrections of the history digest after the numbers critic (2026-11-03)"""
p = "bev_fusion/temp/t16/history_digest.md"
s = open(p, encoding="utf-8").read()
pairs = [
    ("timing 102.8 vs 86.2 ms |", "timing: full network 96.4 ms vs control chain 86.2 ms (102.8 ms was the first small-set run with the radar memory switched on, resolved in the card's reading note) |"),
    ("control `t16.ctl_s0` not run yet (ledger: blocked; run.json: waiting)", "control `t16.ctl_s0` not run (ledger: held by Mirko's order)"),
    ("The ledger shows `t16.ctl_s0` as \"blocked\" in LEDGER.md and as \"waiting\" in `records/runs/t16.ctl_s0/run.json`; the t18.fix run status is \"running\" in its run.js", "At the time of the first reading the ledger showed `t16.ctl_s0` as \"blocked\" and `t18.fix` as \"running\" (now: `t16.ctl_s0` held, `t18.fix` ok); the digest's run.js"),
    ("and radar adds +0.124 / +0.109 (t4_soa_front).", "and radar adds +0.124 / +0.109 over DualBEV-front for the 3000-step arms, +0.133 / +0.117 over the matched camera-only arm (t4_soa_front)."),
    ("Static-cell noise is attributed to the x10 output gain of M under AdamW; the lr is the cheapest lever, the gain the next", "Static-cell noise was attributed to the x10 output gain of M under AdamW and the lr was the cheapest lever to try; the lr (bundled with the hygiene flags) was CUT, the gain lever is untested, and the static error of the large runs is already small (0.19 - 0.24 m/s)"),
]
for a, b in pairs:
    n = s.count(a)
    print(n, a[:60])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
