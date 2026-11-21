# E389b - the E389 route-association head with ONE attention: is the sharp second attention (B) needed, and does a plain attention (no pair-feature bias) read the map at all?

Card of 2026-11-21 (system clock), version 1, written by the report chat on Mirko's order (chat, 2026-11-21 ~00:40, after he read the E389 report: "Sieht sehr kompliziert aus, attention a, attention b... kann man da nicht einfach nur 1x attention nehmen?" and then "nimm normale attention und mache das. Am ende wird die Karte gebraucht, um Routing Informationen zur Verfuegung zu stellen, d.h. die logische Route in der Karte muss den Perzeptionsergebnissen zugeordnet werden."). Variant of E389 (a letter, same task, same data, same read-out, nothing else changed); the verdict of E389 (TIE (NO GAIN)) is NOT re-decided here. The logical road graph of the map (RQ-9, E390 steps 1 - 3) is the other line of work: this card only simplifies the E389 head so that step 2 of E390 starts from the simplest working attention. D88 / D113: no rule step is a bar; D91: one frame, one scan. TEST is never touched.

**The question.** The E389 head has three paths into its output MLP: the token's own features x, the learned cross-attention A (4 heads, none key, bias from the metre-scale pair features) and a sharp, parameter-free soft-argmin attention B (tau 0.3 m) that only passes on the pair features of the nearest route sample. B was added because A's softmax might stay too soft to decide 'within 2 m of the route' in 3000 steps; it was never tested without. Mirko's wish: ONE normal attention. Two variants of the head (`RouteHead(attn=...)`):
- **`a`** = attention A only (B and its pooled features removed): the learned cross-attention with the none key and the metre-scale pair-feature bias, one attention, 33,226 parameters (full inputs).
- **`plain`** = the literal 'normal' cross-attention: logits q.k / sqrt(dh) (+ the none key) and NO pair-feature bias; the pair features reach the output only as the attention-pooled values: 32,806 parameters. Expected to fail at metre-scale registration (q.k of position / 50 cannot compute a 1 m offset); it is run to show whether the bias is the part that matters. Exploratory: no verdict.

## 0. The card (what the tools run)
```json card
{
 "id": "E389b",
 "card": "",
 "title": "E389b: the E389 route-association head with ONE learned cross-attention (arm a: attention A with the metre-scale pair-feature bias and the none key, the sharp attention B removed; arm p: a plain q.k attention without the bias, exploratory), against the E389 head with A and B (m) and the matched no-map head (z) on the same cache, labels, steps and seeds",
 "kind": "block",
 "deterministic": false,
 "metric": "route_F1_road",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.09, "vs": "z"},
 "arms": {
  "a": {"recipe": "RouteHead(attn='a'): one learned cross-attention (A) + the none key + the metre-scale pair-feature bias, no attention B; trained exactly as E389 m (train mode mix, 3000 steps, batch 8, lr 1e-3, bf16 on the GPU); read in the modes error (the record) / exact / none / wrong / gross, paired against E389's z of the same seed", "run_ids": ["E389b.a_s0", "E389b.a_s1", "E389b.read_a_s0", "E389b.read_a_s1"]},
  "p": {"recipe": "RouteHead(attn='plain'): the same without the pair-feature bias (plain q.k attention); seed 0 only, exploratory, no verdict", "run_ids": ["E389b.p_s0", "E389b.read_p_s0"]}
 },
 "control": "m",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "route_F1_road", "vs": "m", "min_effect": -0.01, "note": "by hand: the per-clause rule below (a is read against the stored E389 m and z cards, paired against z of the same seed); decide.py cannot judge it"}
 ],
 "budget_gpu_h": 0.5,
 "kill": "a loss is not finite; or no learning in an arm (the logged `total` of step 3000 (mean of steps 2901 - 3000) is not below 0.8 x the logged `total` of step 100); or the head ignores the map (a: F1_road at the mode wrong within 0.01 of its F1_road at the mode error on BOTH seeds AND a - z < 0.01 at the mode error: then the verdict is 'ONE ATTENTION DOES NOT READ THE MAP' with the status named); or the unit tests of the attn option fail (the code is not used before they are green and a code critic pass is done); `--timeout` as E389 (about 30 min per run, two slots at most)",
 "evaluate": {"split": "val"}
}
```

**Same as E389 (nothing else changes):** cache v4 (route cap 160), the B' token set K 512 of cache seeds 0 / 1, the labels (on_road, on_lane), the delivered tokens (E373 stage-1 head at the D115 decode, p_none < 0.6), the unit (covered GT node), the modes, the bootstrap (paired scene bootstrap, 1000 draws), VAL 873 frames in the E365 cache order, TEST never touched. **References, no rerun:** the stored E389 cards `runs/e389_<tag>/route_card_*.json`: z_s0 / z_s1 (the matched no-map head; the no-map function of the `a` head is the same one-path head, so z of E389 is its matched reference), m_s0 / m_s1 (A + B), the twins z2 / m2 for the noise (|z - z2| = 0.0015, |m - m2| = 0.0021, spread 0.0021). The paired intervals of `a` use the same z dump of the seed as those of m.

**Prediction (guess, written before any run; HIT window +-0.02):** a - z route_F1_road at the mode error = **+0.09** (m - z was +0.0939 / +0.0972): the bias of A carries the registration, B adds nothing the learned A cannot reach. **Odds:** 'ONE ATTENTION SUFFICES' about 60 %, 'B HELPS' about 30 %, inconclusive about 10 %. Arm p (plain): a - p >= +0.05 (the bias is what reads the map) about 70 %.

**The rule (by hand, written before any run; the numbers are those of the score cards at the mode error unless said otherwise, both seeds, paired against z of the same seed):**
- **ONE ATTENTION SUFFICES (adopt the simpler head `attn='a'` as the E389 / E390 starting point; E389's own verdict stays TIE (NO GAIN))** needs all of: (1) a - z >= (m - z) - 0.010 on BOTH seeds (a is not worse than m by more than the noise band; the mean m - z of E389 is +0.0956); (2) trust not worse than m by more than 0.010: a(none) - z >= m(none) - z - 0.010 and a(wrong) - z >= m(wrong) - z - 0.010 on both seeds (the E389 values are in `runs/eval/e389_verdict.json`; this card does not ask for the absolute trust bars E389 failed); (3) F1_road on the fork frames a - z >= (m - z) - 0.010 on both seeds; (4) the kill did not fire.
- **B HELPS (keep both attentions)**: (1) fails on both seeds by more than 0.010 (a is clearly worse than m); the report prints a - m with its paired interval.
- **INCONCLUSIVE:** the seeds disagree (one passes (1), the other fails by more than 0.010), or the E389 spread 0.0021 is exceeded by |a_s0 - a_s1 - (m_s0 - m_s1)| > 0.02: add a seed.
- **Cost, reported:** parameters (33,226 vs 33,738) and the FLOPs of `jroute_eval.py cost` for m and a; timing is NOT taken on this (busy, desktop-loaded) GPU as a verdict (E389: not quiet).
- **Arm p:** numbers reported (F1 road / lane in all modes, a - p, p - z), no verdict. A plain attention that reaches z + 0.05 would be a result in its own right (it would mean the bias is not needed).
- A TIE inside the band means the cheaper (simpler) variant, which is `a`.

## 1. What the task is for (Mirko's purpose)
The map is needed in the end to give the planner ROUTING information: the logical route of the map must be assigned to the perception results (which delivered token lies on the route / on the ego lane). This is the E389 read-out (per covered GT node: on the route's carriageway, on the ego lane). E389b only decides which attention the association head needs; the logical road graph (lanes per direction, connections) is E390, and E390 step 2 (fusion) will start from whatever attention this card leaves as the simplest working one.

## 2. Design delta (rows of DESIGN.md)
DR-5 ('the map is not in the champion: one cross-attention ... (E389)') is attacked in its wording only: E389 had a cross-attention A and the sharp attention B; this card measures the head with A alone and with a plain attention. RQ-9 (the map as a logical road graph, step by step) is not touched; its step 2 reads the outcome of this card.

## 3. Design and runs
| layer | what | where | gate |
|---|---|---|---|
| code | `RouteHead(attn='ab' | 'a' | 'plain')`, trainer `--attn`, the head named in the dump meta and the score card; the default 'ab' is bit for bit the E389 head (33,738 / 21,130 parameters, same state_dict keys) | `training/grid/jroute.py`, `training/train/jroute_train.py`, `evaluation/jroute_eval.py`, `training/grid/test_jroute.py` | tests green incl. the new mutants; a code critic pass |
| train | arms a_s0, a_s1, p_s0 on cache seeds 0 / 1 (init seed = seed), 3000 steps, GPU window only (night), two slots | `run.py --id E389b.<arm> --gpu --require-card` via `temp/e389b/e389b_train.sh` | done flags |
| read | the five modes + paired scores against z of E389 (CPU) | `run.py --id E389b.read_<tag> --require-card` via `temp/e389b/e389b_read.sh` | cards `runs/e389b_<tag>/route_card_*.json` |
| decide | the rule above, by hand, arithmetic on the cards and `runs/eval/e389_verdict.json` | `runs/eval/e389b_verdict.json` | numbers in section 4 |

## 4. Result
(pending)
