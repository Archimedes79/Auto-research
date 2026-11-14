# E338 - stack check (D87 / D88, CPU, no training): the pair heads of E324 s read out on the delivered graph of c2 (B fine-tuned 8250 steps, pyramid frozen) instead of B's: does c2 add to E324 s?

Card of 2026-11-14 04:00 (lane chat, on the lane lead's order: two end-to-end candidates that each stay below or at the adoption bound alone; the stack is only a re-read-out of stored dumps, no retraining). Version 1, written BEFORE any number of the stack is looked at; a cards critic pass precedes the scoring. **Status: card only; the scoring (CPU, niced, about 6 minutes per read-out and seed) starts after the critic.** Only differentiable training is involved (both parts are trained end to end: a longer fine-tune of B with a frozen pyramid, and the E320 recipe with the attached-end head); the output selections of the read-out stay named (D88).

## 0. The card (what the tools run)
```json card
{
 "id": "E338",
 "card": "",
 "title": "E324 s (own keypoints, snapped) on the delivered graph of c2 against E324 s on B's graph, with the same pair heads, both seeds, VAL",
 "kind": "variant",
 "deterministic": true,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.005,
 "prediction": {"value": 0.0043, "vs": "s"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "s": {"recipe": "stored: E324 s own keypoints snapped on B's delivered graph (`runs/e324_s_s<S>/ends_card_snap_thr0.1.json`): topo F1 0.6981 / 0.7010, dup_frac 0.1781 / 0.1522, F1@1.0 0.8762 / 0.8763, junction F1 0.3238 / 0.3232, vertices 18.97 / 19.63, lanes found 0.7604 / 0.7594"},
  "sc": {"recipe": "the SAME pair run `e324_s_s<S>` (maps E307 + pair head; `ends_val.pkl` reused, nothing retrained, nothing re-dumped) with its own keypoints, snapped (`--snap --thr 0.1`), on the delivered graph of c2 read out with the hard step: `jends_eval.py score --run e324_s_s<S> --seed <S> --snap --thr 0.1 --full temp/e292/predfull_e292_c2_s<S>.pkl --base runs/e292_c2_s<S>/model.pt --spec close:1.5@0.5 --tag e338` (the pyramid of the maps run must equal c2's: asserted by the scorer); by hand: replay scoring, no run.py", "run_ids": ["E338.sc_s0", "E338.sc_s1"]}
 },
 "control": "s",
 "variant": "sc",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "s", "min_effect": 0.005},
  {"metric": "F1@1.0", "vs": "s", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "s", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "s", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "s", "max_loss": 0.03}
 ],
 "budget_gpu_h": 0.0,
 "kill": "invalid (not a verdict): the scorer's pyramid assert fails, or the CONTROL does not reproduce: the same tagged scoring of e324_s_s<S> on B's own dump (`--snap --thr 0.1 --full temp/e276/predfull_e273_p_s<S>.pkl --base runs/e273_p_s<S>/model.pt --spec close:1.5@0.5 --tag e338b`) must reproduce the stored card of s (`ends_card_snap_thr0.1.json`) to 1e-6 (the plumbing was verified on E312 s0 on 2026-11-14 with its card of record; this checks it for the E324 s pair run)",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (written before the numbers; by hand: replay of the scored cards, `decide.py` is not used):** **KEEP as a stack candidate** when sc beats s by at least +0.005 topo F1 on BOTH seeds and every guard holds on both seeds (the bounds above; F1@1.0 and junction F1 as losses, dup_frac and vertices as rises). **TIE, kept** when a seed is below +0.005 or a guard is missed but no number breaks the kill: c2 and E324 s stay two separate candidates and the stack is not claimed. There is no CUT by this card (nothing is built or trained by it). 'Beats s' is the stored card of s against the new card of sc, paired scenes (`jdiag.py boot`, interval reported; the 0.005 is a point-estimate bound).

## 1. Card text
- **Statement (falsifiable):** the gain of the longer fine-tune of B (c2: snapped topo F1 +0.0065 / +0.0020 over E312, graph-only +0.0087 / +0.0122) and the gain of the E324 s recipe (own keypoints snapped +0.0156 / +0.0145 over E312) add: E324 s's pair heads read out on c2's delivered graph score at least +0.005 above E324 s on B's graph on both seeds.
- **Mechanism and what could break it:** the connectors of a read-out only depend on B's heads through the delivered pieces they are wired to (the nearest piece end within 4 m and 45 degrees, the long-link cover set) and through the position of those piece ends (the snap draws the curve from the piece end). A fine-tuned graph moves nodes (c2 is B plus 8250 steps of nodes, links, section and assignment heads) and changes which pieces exist; the pair heads of E324 s were trained on B's graph only for the attachment head (`Attach` takes B's raw node outputs); the own-keypoint read-out does not use the attachment (its heads are the maps and the pair logits, which do not read B's nodes: `ends_val.pkl` is independent of B). So the rule reads the OWN-KEYPOINT snapped read-out, the E312 read-out type, on which E324 s is the adoption candidate.
- **Pre-registered numbers (additive guess: sc = s + (c2 - c); a hit is within +-0.003 topo F1 and dup_frac, +-0.01 junction F1, per seed):** topo F1 **0.7046 / 0.7030** (sc - s +0.0065 / +0.0020: the rule's +0.005 is met on seed 0 and expected to be missed on seed 1, so the expected token is TIE); dup_frac 0.1691 / 0.1589; F1@1.0 0.8742 / 0.8792; junction F1 0.3414 / 0.3332; vertices 18.85 / 19.34. Against E312 snapped: topo F1 +0.0221 / +0.0166. The non-additive alternative (the pair heads and the fine-tuned graph interact) is not pre-registered with a number: a deviation of more than 0.003 from the additive value is reported as an interaction, in either direction.
- **Reported, outside the rule (caveat):** the PRIMARY read-out (attached ends, one-piece wiring, `--attach --snap --snap-nogeom`) on c2's graph: the attached ends in `ends_val.pkl` were computed on B's raw nodes, so on c2's graph they point at positions that c2's nodes no longer have exactly; the number is labelled 'attached ends from B's nodes, wired on c2's pieces' and is NOT part of the rule. A faithful test needs a re-dump with c2's node head (GPU, about 5 minutes per run): not done here.
- **Controls and confounds:** s is stored; c2 and c are stored; the same pairs and the same keypoints, only the delivered graph differs (c2's fine-tuned graph with the hard step against B's graph with the hard step); one fine-tune per seed (the scene bootstrap is the interval; retrain noise unknown); the dump of c2 was made by `qgraph_run_eval.py score --dump-full` on the final checkpoint (E292 post_arm). VAL only, TEST untouched.
- **Price and named hard steps:** CPU only, about 6 read-outs of about 6 minutes (niced; the GPU is busy); no new code in a shared file. Named steps unchanged: the close-node step (`close:1.5@0.5`) on the graph, the nearest-piece wiring and the snap geometry, p >= 0.1, the keypoint peak choice, DP in the scorer.
- **Not claimed:** anything on TEST; that the stack is the system of record (the adoption of c2 and of E324 s are decisions of Mirko, together with any TEST use); that the attachment read-out benefits.
