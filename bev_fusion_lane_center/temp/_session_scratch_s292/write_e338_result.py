p = "experiments/E338_stack-check-e324s-on-c2-graph/experiment.md"
s = open(p, encoding="utf-8").read()
a = "## 5. Result\n(not yet: written after the scoring by the rule above)\n"
assert s.count(a) == 1
b = """## 5. Result (2026-11-14 04:45, VAL 873 frames, p >= 0.1, both seeds; numbers from `runs/e324_s_s<S>/ends_card_e338*_*.json` against the stored cards; verdict by `temp/e338/verdict_e338.py`, bootstraps `temp/e338/boot_sc_vs_{s,c}_s<S>.log`, scoring log `temp/e338/score_all.log`)
**Controls (the stop rules) all passed on both seeds:** the tagged scoring on B's own dump (e338b) reproduces the stored card of s in every number and its base block; `curves` of sc equal those of s (11729 / 11880); the base block of sc equals the base block of the stored c2 card.
**Token by the rule written above: TIE, kept (c2 and E324 s stay two separate candidates; the stack is not claimed as KEEP).** Seed 0 meets the KEEP bound (D = +0.0085, interval [+0.0036, +0.0144], guards held), **seed 1 does not (D = +0.0006, interval [-0.0043, +0.0058])**; no seed is below -0.003, so no negative interaction.

| own keypoints, snapped | seed 0 | seed 1 |
|---|---|---|
| s on B's graph (stored) | 0.6981 | 0.7010 |
| **sc on c2's graph (E338)** | **0.7066** | **0.7016** |
| D = sc - s | **+0.0085** | **+0.0006** |
| additive prediction (s + c2 - c) | 0.7046 | 0.7030 |
| sc - additive | +0.0020 (HIT) | -0.0014 (HIT) |
| sc - E312 snapped (c); bootstrap | **+0.0242** [+0.0147, +0.0333] | **+0.0152** [+0.0081, +0.0218] |

**The additive claim holds: every pre-registered number is a hit on both seeds** (hit windows +-0.003 topo F1, dup_frac and F1@1.0, +-0.01 junction F1, +-0.3 vertices): dup_frac 0.1668 / 0.1584 (additive 0.1691 / 0.1588), F1@1.0 0.8737 / 0.8787 (0.8742 / 0.8792), junction F1 0.3418 / 0.3312 (0.3414 / 0.3332), vertices 18.90 / 19.33 (18.85 / 19.34). The expected direction of a miss (below the additive value) did not occur. So the gain of c2 adds to the gain of E324 s exactly as much as c2 has on its own, and c2 has little on seed 1 (+0.0020 over E312): the stack rule fails on seed 1 because c2's own gain there is below the rule's bound, not because of an interaction.
Guards of the rule against s (all held on both seeds): F1@1.0 -0.0025 / +0.0024, dup_frac -0.0113 / +0.0063 (the seed-1 rise uses two thirds of the 0.01 guard), vertices -0.07 / -0.29, junction F1 +0.0180 / +0.0080; lanes found +0.0022 / +0.0086. **Against E312 snapped (c), reporting, not a verdict:** topo F1 +0.0242 / +0.0152, junction F1 +0.0224 / +0.0169 (seed 1 interval [-0.0012, +0.0366]), F1@1.0 +0.0068 / +0.0102, lanes found +0.0251 / +0.0258, dup_frac -0.0079 / +0.0027 (intervals contain 0): the stack of two end-to-end changes (the E320 recipe at 6000 pair steps and the 8250-step fine-tune of B with a frozen pyramid) passes the E312 adoption rule (> +0.01 on both seeds, guards held) with more margin than E324 s alone (+0.0156 / +0.0145). Adoption of c2 as B and of E324 s, and any TEST use, are Mirko's decisions (HANDOVER questions); nothing adopted, nothing on TEST.
**Caveat read-outs (outside the rule; the attached ends come from B's nodes, wired on c2's pieces):** PRIMARY (attached ends, snap-nogeom wiring) on c2's graph 0.6989 / 0.6949 against 0.6912 / 0.6948 on B's graph (+0.0077 / +0.0001), dup_frac 0.1777 / 0.1655 (B: 0.1897 / 0.1592); own keypoints snap-nogeom on c2's graph 0.6960 / 0.6925 (B: 0.6888 / 0.6916). The geometry share of the snap on the new graph stays: own snapped minus own nogeom +0.0106 / +0.0091, minus PRIMARY +0.0077 / +0.0067. The attached read-out gains from c2 on seed 0 only; a faithful test needs a re-dump with c2's node head (GPU, not done).
**Named hard steps (D88), unchanged:** the close-node step on c2's graph, the nearest-piece wiring and the snap geometry, p >= 0.1, the keypoint peak choice, Douglas-Peucker. Price: none at inference beyond the two parts (c2 costs no ms, E324 s's attachment head is not needed by the own-keypoint read-out).
"""
open(p, "w", encoding="utf-8", newline="\n").write(s.replace(a, b))
print("ok")
