# E363 - the end-to-end STACK on faithful graphs: the pair run of E324 s re-dumped with the node head of B' (E292 c2 plain / a with the Matrix-NMS layer), read out at the end-to-end read-out with the rule wiring and with the learned wiring head of E358

Card of 2026-11-16 13:55 (system clock), version 1. Written by the lane experiments chat on the lane lead's order (D109 / D111: the end-to-end stack as one system; E350 v3 = E358 and E292 = E360 as the two candidates), as the follow-up of both: **E360** (the Matrix-NMS layer against its end-to-end control, token INVALID by the pre-registered symmetry stop rule because the stored attached ends were computed on B's nodes, while the graph under them was B''s; would have been TIE) and **E358** (the wiring head, KEEP+ at the end-to-end read-out on B's graph). Here the attachment is recomputed by the node head of the graph it is read out on (checkpoint surgery of the pair run, run folders `runs/e363_<c2|a>_s<S>`, `temp/e363/make_run.py`), so the asymmetry of E360 is gone by construction. **Status: card v1, the four faithful dumps exist (`E363.dump_*`, data preparation, `run.py --gpu --no-card`), nothing scored; the first critic pass is the next step (section 4).** CPU only from here on (no GPU, no training).

**What the outcome decides.** (1) Layer against its end-to-end control at the faithful read-out (arms a, c): KEEP = the layer is kept as the end-to-end CANDIDATE for R2 in the stack (nothing adopted into the system of record: that is the lead's / Mirko's decision, D110 stays an open question); TIE = the layer stays the documented differentiable alternative (D87). (2) The learned wiring head on those graphs (arms aw, cw: a TRANSFER of the E358 head, trained on B's graph records): reported with the sign class of E358's rule against its own graph (aw against a); this number is the stacked system's end-to-end read-out. No token is a CUT unless the kill of E358 / E360 applies (there is no training here).

## 0. The card (what the tools run)
```json card
{
 "id": "E363",
 "card": "",
 "title": "E324 s (the pair run re-dumped with the node head of B': `runs/e363_<g>_s<S>`) read out at the end-to-end read-out (`--attach --snap --snap-nogeom`, p >= 0.1, `--spec plain@0.5`) on the graph of B' with the Matrix-NMS layer (E292 a) against the graph of B' without layer and without the hard step (E292 c2), both with the attachment recomputed on their own node head",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.01,
 "prediction": {"value": 0.020, "vs": "c"},
 "better": {"dup_frac": "lower", "vert100_dp": "lower"},
 "arms": {
  "c": {"recipe": "faithful end-to-end control: the pair run with the node head of c2 (B fine-tuned, pyramid frozen, NO layer, NO hard step): `bash tools/env/_py14.sh jends_eval.py score --run e363_c2_s<S> --seed <S> --thr 0.1 --attach --snap --snap-nogeom --full <root>/temp/e292/predfull_e292_c2_s<S>.pkl --base <root>/runs/e292_c2_s<S>/model.pt --spec plain@0.5 --tag e363c`", "run_ids": ["E363.c_s0", "E363.c_s1"]},
  "a": {"recipe": "the layer arm: the pair run with the node head of a, the graph of a (Matrix-NMS layer, no hard step): the same command with `--run e363_a_s<S>`, `--full <root>/temp/e292/predfull_e292_a_s<S>.pkl --base <root>/runs/e292_a_s<S>/model.pt`, `--tag e363a`", "run_ids": ["E363.a_s0", "E363.a_s1"]},
  "cw": {"recipe": "stack control (reported): arm c with the E358 head (transfer; head 0 of `runs/e358_w_s<S>`) on the STAGED scorer: `bash temp/e358/run_staged.sh jends_eval.py score ...` as c plus `--wire head --wire-run e358_w_s<S> --tag e363cw`", "run_ids": ["E363.cw_s0", "E363.cw_s1"]},
  "aw": {"recipe": "THE STACK: arm a with the E358 head: as cw with `e363_a_s<S>` and `--tag e363aw`", "run_ids": ["E363.aw_s0", "E363.aw_s1"]}
 },
 "control": "c",
 "variant": "a",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "c", "max_loss": 0.005},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01},
  {"metric": "dup_frac", "vs": "c", "max_loss": 0.01},
  {"metric": "vert100_dp", "vs": "c", "max_loss": 1.0},
  {"metric": "junction_F1", "vs": "c", "max_loss": 0.01},
  {"metric": "lanes_found", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 0.0,
 "kill": "the replay is INVALID (nothing is cut; the card is re-run after the fix) when a STOP RULE fails: (i) the WHOLE `numbers.base` block of c and a equals the stored graph-level cards `runs/e292/e292_{c2,a}_s<S>.plain_t0.5/card.json` (the same B' graphs as in E360); (ii) `curves` equal the stored 11729 / 11880 (the pair logits do not depend on the node head); (iii) the FIXTURE fails: the dump of the pair run with B's OWN node head (`runs/e363_b_s0`, 0 tensors changed, `E363.dump_b_s0`) does not reproduce the stored `runs/e324_s_s0/ends_val.pkl` (every array of every frame allclose, atol 2e-3: the dump pipeline and the surgery are then not the stored ones); (iv) the scorer's pyramid check fails or a head provenance check fails; (v) `E363.dump_*` is not complete (873 frames). The symmetry stop rule of E360 is NOT used here: its reason (attached ends computed on another graph's nodes) is gone by construction; the shares of attached ends within 1 m of a delivered piece end are REPORTED. There is no training: no loss to be non-finite",
 "evaluate": {"split": "val"}
}
```
**Verdict rule (written before any scoring; the scoring jobs go through `run.py` as CPU jobs with `--retries 0`; the verdict is by hand: `temp/e363/verdict_e363.py` written and self-tested BEFORE the first scoring; label 'by hand: D87 replay'):** per seed the table of the six metrics a - c with the bounds of the JSON (all against c). **KEEP (the layer is kept as the end-to-end candidate for R2; nothing adopted)** only when on BOTH seeds topo F1 of a is at least c - 0.005, every guard holds, the paired scene bootstrap of a - c (`jdiag.py boot`, topo F1) has its lower bound above 0 (KEEP+), and no stop rule failed. **TIE, kept and refined (D82 / D87)** otherwise (a bound or guard missed, an interval containing 0 or negative). INVALID when a stop rule fails. **Expected token: TIE** (the duplicate guard: +0.0178 / +0.0122 in E360 at the stale read-out and +0.0185 / +0.0097 at its own-keypoint witness, the faithful read-out sits between). **Prediction, per seed, with hit windows:** a - c topo F1 **+0.016 (seed 0) / +0.024 (seed 1)** (the values of E360's PRIMARY: +0.0162 / +0.0244; its own-keypoint witness, which does not depend on B's nodes: +0.0162 / +0.0239), HIT = within +-0.008; dup_frac a - c +0.015 / +0.010 (+-0.006); junction F1 a - c +0.085 / +0.100 (+-0.030); lanes found a - c -0.002 / -0.004 (+-0.010). **Stack, reported with its own prediction (E358's result carried over): aw - a = +0.0019 / +0.0016 topo F1 (hit |aw - a - that| <= 0.003, i.e. the head gains about as much on the layer graph as on B's graph), guards as E358 (against a, `max_loss` 0.005 / 0.01 / 0.01 / 1.0 / 0.01 / 0.01) and its sign class by the paired bootstrap (`jdiag.py boot` of aw against a); cw - c is its control.** Reported next to the rule, not in it: the recovery share (a - c) / (E338a - c) (E338a: c2 WITH the hard step, 0.6989 / 0.6949, attached ends from B's nodes); the stack against E324 s PRIMARY on B with the step (0.6912 / 0.6948) and against E358 w (0.6931 / 0.6964); dup_frac against E312 snapped; the shares of attached ends within 1 m; the number of delivered curves; ms NOT measured (the layer has no parameters, the head 545; D106 'simpler AND faster?' not answered). The old tokens of E292, E358, E360 stay as they are in every case. D110 stays open: nothing is called 'adopted'.

## 1. Card text
- **Statement:** with the attachment recomputed on each graph's own node head, the Matrix-NMS layer still gains over the end-to-end graph without any duplicate suppression, at the end-to-end read-out, by about what E360 measured with the stale attachment and what its own-keypoint witness measured; and the E358 wiring head transfers to the layer graph with about its own gain.
- **Mechanism:** the keypoints and the pair logits come from the frozen pyramid, the maps head and the pair head (identical for all arms); the attached ends are the soft attachment of those keypoints to the nodes of the graph's own node head (`Attach`, `jends.py:97-136`; B's node head is a frozen module of the pair model, `jends_train.py:33`), so swapping `nodes.*` (38 tensors) of the pair model for those of B' gives the pair run that would have been trained against B' nodes, up to the attachment weights (trained on B's nodes: a named caveat). Surgery and dump are checked by the fixture (iii).
- **Inputs (ready):** `runs/e363_{c2,a}_s{0,1}/` (model.pt with the swapped nodes, `provenance.json`, `ends_val.pkl` from `E363.dump_*`), `temp/e292/predfull_e292_{c2,a}_s{0,1}.pkl` (B' graph dumps, scratch, untagged), `runs/e292_{c2,a}_s{0,1}/model.pt`, `runs/e358_w_s{0,1}/` (heads), the stored cards for the stop rules; the scorer `jends_eval.py score` unchanged for c, a; the STAGED scorer of E358 (`temp/_session_scratch_e358/stage`, with the derived-run check of `check_head`: a head trained for the pair run P may be read on a run folder whose `provenance.json` names P as `source_pair_run`: tests and mutants in `temp/_session_scratch_e358/test_staged.py`) for cw, aw.
- **Starts from:** E292 (a, c2), E324 s (pair run), E358 (head), E360 (the stale-attachment version of this comparison).
- **Not claimed:** anything on TEST; adoption into the system of record; ms; that the attachment weights (trained on B's nodes) are optimal for B''s; that the stack is end-to-end TRAINED (the head is learned by imitation, the layer is trained in B''s fine-tune, the pair run is not re-trained on B' nodes: a joint training is not done).
- **Named hard steps that remain:** the layer's read-out threshold e >= 0.5 (T1), the wiring argmax and its candidate gate and the long-link cover drop, the keypoint peak choice and top-K, p >= 0.1, the rest of R1 - R8, T1 - T3 of the gap table; the close-node step is the one that leaves in arms a / aw.
- **Controls:** c (no layer, no step, same dump procedure) is the end-to-end control; cw is the head's control on c; the fixture proves the surgery and the dump; the stored cards of E358 / E360 are the references; the scene bootstrap is the interval; each B' is one fine-tune draw per seed.
- **Price and timeline:** CPU only, **8 scorings** (c, a, cw, aw on both seeds), one or two at a time (nice, 2 threads each), about 10 minutes each: scoring 14:30 - 15:45, result section, ledger notes and commit by 16:15; nothing is left running at the shutdown (17:30, D111; no GPU is used by this card after the dumps).
- **Referee:** none yet (section 4).

## 2. What exists and what has to be built
Exists: the four faithful run folders and dumps (`temp/e363/make_run.py`, `start_dumps.sh`; the fixture folder `runs/e363_b_s0` and its dump `E363.dump_b_s0`), the scorer options for c, a (`--full`, `--base`, `--spec`, `--attach`, `--snap-nogeom`, `--tag`; tags `e363c`, `e363a`, `e363cw`, `e363aw`), the staged scorer with the head provenance, `run_staged.sh`. To build before the first scoring: `temp/e363/verdict_e363.py` (the table, the stop rules (i) - (v), the bounds, the bootstrap intervals from `jdiag.py boot` logs, the stack lines, the token; selftest with planted cards), `temp/e363/fixture.py` (the dump comparison of stop rule (iii)), `temp/e363/start_chain.sh` (the eight `run.py` jobs, `--retries 0`, `--after` chains per seed; 'dry' by ARGUMENT, not by environment variable). **Order:** fixture and verdict script first, then per seed c, a, cw, aw.

## 3. Risks
(1) The attachment weights were trained on B's nodes: on B''s nodes they are out of distribution (the fixture does not test that); the result is for 'the pair run read on B''s nodes', named. (2) c2 plain's duplicates are wired by the connectors (E360 saw dup_frac +0.0178 / +0.0122 for the layer): TIE expected. (3) The head is a transfer: it was trained on B's graph with the hard step (piece attributes of B's plain graph at e >= 0.5 after `close:1.5@0.5`); on c2 plain / a the pieces differ (more or fewer fragments): the head may lose; arm cw tells. (4) Two fine-tune draws (c2, a) per seed. (5) The scorer refuses to overwrite a tagged card: one tag per scoring, `--retries 0`. (6) WSL page cache (OSError 12 hit E360): the cache guard runs until 17:30. (7) The B' dumps are untagged scratch.

## 4. Referee
(empty: the first fresh ar-critic pass, mode cards, is the next step)

## 5. Build log
(empty)

## 6. Summary
(empty until the runs exist)
