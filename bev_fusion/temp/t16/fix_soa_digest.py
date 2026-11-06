"""scratch: corrections of the SOA digest after the numbers critic (2026-11-03): the re-scoring on GT v8 exists now (section 6.3 of the report)"""
p = "bev_fusion/temp/t16/soa_digest.md"
s = open(p, encoding="utf-8").read()
pairs = [
    ("| not found: never run on `t16_full_s0` / `t17_data` | not found |", "| run on 2026-11-02 (`temp/t16/soa/front_v8_n1000.md`, report section 6.3): DualBEV mAP2 0.245, `t16_full_s0` 0.442 | not found |"),
    ("A fair SOA claim on the t16 network needs one new experiment: re-score", "That experiment was done on 2026-11-02 (report section 6.3): re-score"),
    ("- No SOA model has been scored on GT v8, the fusion-area scoring or the t16 / t17 frames; all stored re-scorings use GT older than v6 (S1-S5).", "- (closed 2026-11-02, report section 6.3) GaussianCaR's cached logits and the front-only DualBEV port were re-scored on GT v8 next to `t16_full_s0` (maps: 384 frames; front sector: 1000 frames); the stored re-scorings S1-S5 use GT older than v6."),
    ("- No front-sector mAP2 (`bev_suite.py front`) of `t16_full_s0` or `t17_data`.", "- (closed 2026-11-02) the front-sector mAP2 of `t16_full_s0` is in `temp/t16/soa/front_v8_n1000.md`; the CRN comparison is still open."),
]
for a, b in pairs:
    print(s.count(a), a[:70])
    s = s.replace(a, b)
open(p, "w", encoding="utf-8").write(s)
