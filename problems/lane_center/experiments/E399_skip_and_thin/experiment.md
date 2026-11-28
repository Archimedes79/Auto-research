# E399 a soft skip-link layer and fewer tokens, on the champion's weights

The card is the json block below: `research check <problem> E399` checks it, `research start` freezes it in the snapshot. What the runs do,
how the verdict is made and what counts is the card as the snapshot holds it; a changed experiment is a new experiment.

```json card
{
 "id": "E399",
 "title": "a soft skip-link layer and fewer tokens on the champion's weights",
 "question": "on the champion's weights, without training: does a soft layer that damps links another path already covers, token thinning by existence (above 0.2), or both raise topo F1 on VAL without breaking a guard?",
 "hypothesis": "59.5 % of the forks hold a jump that another branch reaches anyway; damping such links removes false forks and duplicates and raises topo and junction F1 at a small loss of lanes found; the weak tokens (existence at most 0.2) add little but noise, so leaving them out cuts a third of the tokens without loss",
 "prediction": 0.018,
 "kind": "compare",
 "control": "champ",
 "seeds": [0, 1],
 "arms": {
  "champ": {"from": "E396.champ", "note": "the champion's stored runs"},
  "skip": {"set": {"import_weights": "runs/e399_skip_s{seed}"}, "note": "the champion's weights, the soft skip-link layer on the final wiring (sigma 0.03, paths of 2 to 5 links over the 8 strongest links, the read-out's thresholds)"},
  "thin": {"set": {"import_weights": "runs/e399_thin_s{seed}"}, "note": "the champion's weights, only tokens with existence above 0.2 take part"},
  "both": {"set": {"import_weights": "runs/e399_both_s{seed}"}, "note": "both changes"}
 },
 "rule": {},
 "kill": [],
 "note": "asked for by the owner on 2026-11-28 (skip links, fewer than 512 nodes, iterate); the decode stays the champion's (0.3 / 0.2 / 0.6); no adoption without the owner"
}
```

## Why (champion, VAL, E398 follow-up diagnostics; scratch scripts and `tools/diagnose`, no record)
- 59.5 % of the delivered forks hold a link whose target another branch of the fork reaches within 4 links (3.6 % of the links, 73 % of
  them long-class, median chord 13 m). Removing those at the read-out (a rule) raised topo F1 by 0.023 / 0.025 (seeds 0 / 1).
- Stage check: outside intersections the pixel stage covers 96 % of the true lane points, confident nodes 88 %, kept nodes 94 %, delivered
  lines 95 %; inside intersections delivered lines 83 % (no node targets there by design). Of 442 valid tokens per frame 167 have existence
  0.5 or more.
- A first remedy (the long class's own none weight in training, drafted as this experiment before it was started) was dropped after a check on
  TRAIN: its read-out equivalent gave +0.0002 / -0.005 and higher long thresholds only lose; the critic's findings are in that decision.

## Chosen on TRAIN (the champion's TRAIN sample, every third frame, 291 frames; seeds 0 / 1)
| variant | tokens per frame | topo F1 | junction F1 | lanes found |
|---|---|---|---|---|
| champion | 448 / 443 | 0.7192 / 0.7225 | 0.3399 / 0.3396 | 0.8095 / 0.8019 |
| skip layer (numpy form) | 448 / 443 | 0.7364 / 0.7414 | 0.3908 / 0.3821 | 0.7956 / 0.7903 |
| existence above 0.2 | 301 / 306 | 0.7254 / 0.7256 | 0.3479 / 0.3458 | 0.8005 / 0.8012 |
| both | 301 / 306 | 0.7400 / 0.7438 | 0.3970 / 0.3936 | 0.7876 / 0.7876 |

The skip layer's form (hard removal; max-product over soft strengths with a sigmoid of width 0.1 / 0.03; ramps) and the thinning rule (top K
384 / 320 / 256 / 192 / 128; existence above 0.1 / 0.2 / 0.35; near-duplicate removal) were chosen among these variants on TRAIN: sigma 0.03
came closest to the hard removal (0.7414 / 0.7449), existence above 0.2 saves a third of the tokens inside the lanes-found guard (0.35 saves
half, with lanes found 0.04 lower).

## The layer
`net/model/jtoken.skip_suppress`, no parameter, differentiable: the strength of a link is a sigmoid of (probability - threshold of its class)
times that of both ends being kept; R(i, j) is the largest product of strengths over a path of 2 to 5 links; the probability of i -> j becomes
p (1 - R), the removed mass goes to the source's none slot. The gradient passes only near the thresholds. Thinning: `JTokenHead(min_e)`: a
token with sigmoid(existence) at most 0.2 is invalid (no attention, no candidate) and reads as none. Both are options of the head's args,
off by default (bit for bit the champion).

## Imported networks
`<runs root>/lane_center/e399_{skip,thin,both}_s{0,1}`: the champion's `e372_s4_s{0,1}/model.pt` with the option added to its args, weights
and B' tensors equal (checked); `import.json` in each folder records both sha256.

## How the verdict is read
The standard rule per arm. KEEP: a candidate; the owner decides an adoption. The layer is a fixed soft step in the forward path, like the
soft Matrix-NMS layer (E292); whether it satisfies R1 as it stands, or must be learned, is the owner's call at an adoption. Lanes found may
drop (TRAIN -0.014 for the layer): the guard allows 0.010.

## Design delta
R1: a second fixed soft layer (on the wiring). R5: the token set can be thinned by existence below the budget of 512. DESIGN.md changes only
at an adoption.

## Plan
- Code on branch `lane/skip-links` (worktree `temp/wt_skip`): `jtoken.skip_suppress`, `JTokenHead(skip=, min_e=)`, `build_head` reads
  `skip_suppress` and `min_e`; tests `test_skip_suppress`, `test_min_e`; `tools/diagnose` (one command per run, one per experiment).
- Checks before the start: `test_jtoken` full, layering, drawing, anchor of `E396.champ_s0`.
- Runs: predict and evaluate only (6 runs, about 5 minutes each, 2 at a time). TEST is not touched.

## Conclusion
**Verdicts** (`research judge`, 2026-11-28, `verdict.json`): skip TIE, thin TIE, both TIE; none adopted. All numbers VAL (873 frames, 22
scenes, scene bootstrap), pooled over seeds 0 / 1, against the champion's stored runs (topo F1 0.6748). TEST not touched.

| arm | topo F1 gain [95 %] | junction F1 (0.2510) | lanes found (0.7223) | F1@1 (0.8467) | duplicates (0.0637) |
|---|---|---|---|---|---|
| skip | +0.0185 [+0.0132, +0.0245] | 0.2882 | 0.7058, guard broken | 0.8396 | 0.0183 |
| thin | +0.0040 [+0.0020, +0.0065] | 0.2553 | 0.7152 | 0.8446 | 0.0566 |
| both | +0.0213 [+0.0158, +0.0274] | 0.2976 | 0.6996, guard broken | 0.8376 | 0.0160 |

- Prediction 0.018: met by skip and both, not by thin.
- `tools/diagnose` per run: forks per frame 10.2 / 11.0 with thinning alone, 4.6 / 4.7 with both; share of forks with a jump 59 / 60 % with
  thinning alone, 3 - 4 % with the skip layer.
- Learned: the skip layer removes the jumps it was built for and most duplicates, and gains topo F1 on VAL as on TRAIN; it costs lanes found
  beyond the guard. Thinning alone is close to neutral (a third of the tokens fewer on TRAIN).
- Follows: E420 adds a soft gap-closing layer on top of both, against the missing links.
