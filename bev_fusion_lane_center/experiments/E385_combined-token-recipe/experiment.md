# E385 - the combined token recipe: the champion plus ONLY what survived its own written rule, confirmed together on the seed-twin head seeds of the parts

Card of 2026-11-18 (system clock). **Version 2 at 02:40, after the first critic pass** (fresh ar-critic, cards: REVISE, G2 no, 6 blocking + 12 should-fix; taken in: (B1) the hold rule now also holds the combination against the best adopted single part, 'adds up' is measured against it, and D106 picks the simplest subset inside the band; (B2) the plain trainers only (the speed path does not cover KD / S4), e2 out of E385, KD and S4 never together (G-simple), the teachers when max-positive is also taken decided by a CPU check now; (B3) a composition deadline: any status other than the adopt verdict at 10:30 = not taken, E384 excluded; (B4) c on head seeds 2 / 3 = the seed twins of cb, mx, d and s4 (a 2 x 2 at no extra GPU cost); should-fix: run ids, both stage commands, the verdict cell, kills per stage, the cost that counts (the delivered path, paired timing), K 384's reason, the robustness read-out, the schedule, the budget per composition, G-simple / G-cost and the labels, a launch check on the simplified code, the odds). Version 1 (02:05, 65e8b2fe): a draft with an open composition. Written by the Spurmitte chat that Mirko addresses as "Fable" on the lane lead's order of 01:55 (D117 point 4 / D119: "ONE combined recipe card ... the rule written before any run, two seeds, VAL, a fresh ar-critic pass"). Mirko, D119 (00:25): "bring this system to a finish line. Goal E2E differentiable. Fast. Simple & elegant, robust, good performing." D113 / D109 / D55 (E2E differentiable only, no rule step; at a tie the end-to-end variant, D109), D106 (the same cost: a tie takes the plain recipe), G-simple (E365: at most two training stages), the six hard rules. TEST never touched.

**The question:** do the gain parts that passed their own cards (max-positive E378, distillation E379), trained TOGETHER (and with S4 if it holds), keep or add up their gains on the seed-twin heads, with no rule step and at most two training stages - i.e. what is the finished one-head system of D119?

## The composition (decided by this table from the parts' own written verdicts as they stand at 10:30 on 2026-11-18; written before any run; any status other than the adopt verdict at 10:30 - INCONCLUSIVE, provisional, waiting, not read - means NOT taken)
| part | taken when (by 10:30) | when not taken | notes |
|---|---|---|---|
| base | always: E365 head, none weight 0.25 (E373), the D115 per-class decode (E375), K 512 | - | the champion: VAL topo F1 0.6679 / 0.6762, junction F1 0.2517 / 0.2452 |
| MP = max-positive loss (`--pos-loss max`) | E378 ADOPT-LOSS | `--pos-loss sum` | E378 needs x_s1 and its twins c; its seed-twin control head for E385 is E376's mx (head seeds 2 / 3, the same recipe as E378's x) |
| KD = distillation (`--distill-from`, W 1.0) | E379 ADOPT-DISTILL AND S4 not taken (G-simple: teachers + student + S4 would be three stages; S4 has precedence for D119's 'Goal E2E') | no teacher | teachers: if MP is not taken: E379's (e373_cb_s<S>, e367_b25_s<S>); if MP is taken: the CPU check below decides |
| S4 (B''s node head trained with the head's loss) | E372 S4-HOLDS | stage 1 only ('staged, B' frozen') | stage 2 after the head stage (`jlive_train.py --head-run e385_c_s<S>`, E372's s4 settings) |
| e2 (the plain 2-head ensemble) | never in E385 (two heads at inference; e2 + S4 is undefined; e2 with MP would need four trainings); e2 stays the lead's separate D106 decision on E379's time gate | - | - |
| speed path (E383) | not a training part of E385: c trains on the plain trainers every part's verdict was measured with; the speed path is reported as the cost half under E383's own parity verdict | - | the speed path does not cover KD (GPU loader) or S4 |
| K 384 | not taken: E365's K rule (coverage of GT nodes at K 384 = 0.9878 < 0.99 -> K 512); a change of K needs its own card (owner: the report chat, E383's diagnostic) | K 512 | - |
| near-node none weight | never (E377 TARGETING-ONLY, closed; removed from the code by D117) | - | - |
| recursive temporal fusion | never in E385 (E384 cannot be decided today: no critic, nothing built) | - | the next card after E385 |

**MP and KD both taken (the teachers):** a CPU check run now (02:40, before any verdict, `temp/e385/maxens.sh`: `jtoken_eval.py ensemble --runs e376_mx_s0,e378_x_s0 --seed 0 --ref-run e376_mx_s0` at D115, VAL): if the ensemble of two max heads beats one max head by at least +0.003 topo F1 (the head-seed spread), KD is taken with MAX teachers (e376_mx_s<S> + e378_x_s<S>; no verdict of their own: labelled so); else MP and KD are substitutes and only MP is taken (D106: no teacher stage). The check's number goes into section 3 before the composition is fixed.
**When E385 runs at all (the seed twins make most compositions already measured):** a single taken part needs no run (MP only = E376's mx on the same seeds and recipe; KD only = E379's d; S4 only = E372's s4: their own verdicts decide); nothing taken = the champion stays. E385 trains only for (a) MP + KD with max teachers (stage 1 c_s0 / c_s1, the KD flags) and (b) MP + S4 (no new stage 1: stage 2 = S4 on the existing max heads, `--head-run e376_mx_s<S>`, read live against e372_s4_s<S> and against e376_mx_s<S>); KD + S4 never (G-simple).

## 0. The card (what the tools run)
```json card
{
 "id": "E385",
 "card": "",
 "title": "E385 combined token recipe: the champion plus the gain parts taken by the composition table (max-positive E378, distillation E379; S4 E372 as a second stage if it holds), trained on head seeds 2 / 3 (the seed twins of e373_cb, e376_mx, e379_d, e372_s4), read at the verdict cell against the champion and against the best adopted single part",
 "kind": "variant",
 "deterministic": false,
 "metric": "topo_F1",
 "direction": "higher",
 "band": 0.005,
 "prediction": {"value": 0.005, "vs": "ref"},
 "arms": {
  "ref": {"recipe": "the champion: the E373 heads `runs/e373_cb_s0 / s1` (head seeds 2 / 3) at the D115 cell: VAL topo F1 0.6679 / 0.6762, F1@1.0 0.8459 / 0.8467, lanes found 0.7191 / 0.7205, junction F1 0.2517 / 0.2452; read live it equals its cache read exactly (tokens_exact 873 / 873, both seeds, E372's parity read)"},
  "c": {"recipe": "stage 1: `python3 tools/jarm.py run E385 c_s<S> --seed S --head-seed H --ref e373_cb_s<S> -- --none-weight 0.25 [--pos-loss max] [--distill-from <teachers> --distill-weight 1.0]` (S = 0 / 1, H = 2 / 3; the flags by the composition table); stage 2 only if S4 is taken: `python3 ../autoresearch/tools/run.py --id E385.s4_c_s<S> --seed S --gpu --ram-mb 8000 --require-card --timeout 90 ... -- bash tools/env/_pyr.sh jlive_train.py --name e385_s4_c_s<S> --seed S --head-run e385_c_s<S> --train-nodes 1 --live-inputs 1 --steps 3000 --bs 8 --lr 1e-4 --workers 4 [--pos-loss max]`, then `jtoken_eval.py dump --live` / `score --live` (E372's read-out); the exact lines written in section 3 when the composition is fixed", "run_ids": ["E385.c_s0", "E385.dump_val_c_s0", "E385.dump_train_c_s0", "E385.sweep_c_s0", "E385.choose_c_s0", "E385.cell_vs_c_s0", "E385.d115_vs_c_s0", "E385.c_s1", "E385.dump_val_c_s1", "E385.dump_train_c_s1", "E385.sweep_c_s1", "E385.choose_c_s1", "E385.cell_vs_c_s1", "E385.d115_vs_c_s1", "E385.s4_c_s0", "E385.dump_live_s4_c_s0", "E385.read_s4_c_s0", "E385.s4_c_s1", "E385.dump_live_s4_c_s1", "E385.read_s4_c_s1", "E385.vs_part_s0", "E385.vs_part_s1", "E385.drop_s0", "E385.drop_s1", "E385.timing"]}
 },
 "control": "ref",
 "variant": "c",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "topo_F1", "vs": "ref", "max_loss": 0.005},
  {"metric": "junction_F1", "vs": "ref", "max_loss": 0.01},
  {"metric": "F1@1.0", "vs": "ref", "max_loss": 0.005},
  {"metric": "lanes_found", "vs": "ref", "max_loss": 0.01}
 ],
 "budget_gpu_h": 2.5,
 "kill": "per stage, the taken parts' own kills re-stated for this recipe: stage 1: a loss not finite; no learning (the mean head loss over steps 1801 - 2000 not below 0.8 x the mean over steps 101 - 200); the twin check (head seeds 2 / 3: the first batch's frames = E373's order, the step-0 init hash = E373 cb's: proven on the simplified code before the run); with KD: E379's teacher start check (the 20-batch mean of each teacher's GT loss in a band RE-MEASURED on CPU with the student's loss options before the run when MP is taken, written in section 3) and the logged KD over the last 100 steps in a band re-measured the same way; stage 2 (S4): E372's s4 kills (live_inputs asserted, the node-head gradient > 0 at step 1, LabelOverflow <= 1 %, the B-loss rise 1.2 x, the head-loss rise 1.10 x); `--timeout` 90 min per stage (jarm's fixed timeout; the slowest stage-1 variant measured: E379.d 70 min shared); one family on one slot with a supervisor and resume (a resume breaks the twin and is reported)",
 "evaluate": {"split": "val"}
}
```
**The verdict cell:** D115 (the champion's; E378 x_s0, E372 and E383 read at D115), unless a taken part's own verdict moved the recipe's cell by its own rule (then that cell, named in section 3 before the run); the cell `choose-cell --runs e385_c_s0,e385_c_s1` picks on TRAIN is reported. **The read-outs (named before any run; the standard scorer, all sections 0, no rule step, VAL 873, TEST never touched; decide.py is not the verdict):** per head at the verdict cell: c minus the champion (`--ref-run e373_cb_s<S>`, cross-run paired interval), c minus the best adopted single part's twin head (`--ref-run e376_mx_s<S>` for MP, `e379_d_s<S>` for KD; E385.vs_part_s<S>), with S4 taken the LIVE read of stage 2 against E372's s4 heads and against the champion read live (the parity of E372 holds: exact) and stage 1 read on the cache (S4's share visible); the degraded read-out `--drop 0.17` for c and for the champion at the verdict cell (E385.drop_s<S>; on the cache only: `--live` refuses `--drop`, a named gap for a stage-2 recipe); the TRAIN-to-VAL drop, dup_frac, vertices per 100 m; the anatomy; the delivered-path cost: the state-dict shapes of c equal the champion's (asserted: no taken part changes the architecture) and the whole-path ms of c and the champion measured paired in ONE quiet session (A B B A, `jtoken_eval.py timing`, the load stated; E385.timing).
**The verdict (by hand: written before any run):** **ADOPT-RECIPE** (the finished one-head system of D119) = on BOTH heads at the verdict cell: (i) against the champion: topo F1 not lower by more than 0.005, junction F1 not lower by more than 0.010, F1@1.0 not lower by more than 0.005, lanes found not lower by more than 0.010; (ii) against the best adopted single part's seed-twin head: topo F1 not lower by more than 0.003 (the head-seed spread: the combination must not lose what a part alone won); (iii) the D113 gate floors; G-simple (at most two training stages); G-cost (the same architecture: the whole path's ms within the paired timing's noise of the champion's, at most 19.7 ms); no rule step; the kills did not fire; the launch check passed. **Inside the band, the simplest subset (D106):** if c is within +-0.003 topo F1 of the best single part on both heads, the recipe is the SIMPLER of the two in this order: the champion < + MP (a loss change) < + S4 (a second stage; preferred at a tie by D109 / D119 'Goal E2E') < + KD (a teacher stage and two teacher heads); i.e. c is adopted over the single part only when it adds something or when the extra part is S4. **ADDS-UP** (reported with ADOPT-RECIPE): c minus the best single part >= +0.003 topo F1 on both heads. **NO** otherwise: the recipe falls back to the best single part that passed its own card (or the champion); the 2 x 2 of seed twins (cb, mx, d, c on the same head seeds) attributes the failure to a part. The adopted recipe becomes the champion's line in D114 through the lead (a new D<n>), with its label: S4 taken = 'end to end: the loss reaches B''s node head'; not taken = 'staged, B' frozen' (the named gap of D119's 'Goal E2E').
**Prediction (per composition, guesses; refined at the second look):** MP only: c = E378's recipe on E376's seeds = E376's mx (c - mx about 0: no new information: then E385 is replaced by E376 / E378's verdicts and does not run); MP + KD (max teachers): c - mx +0.002 (window -0.003 to +0.007), c - cb +0.008; KD only: c = E379's d (no run, the same argument); MP + S4: c - mx about 0, c - cb +0.006 live. Odds for a run that happens: ADOPT-RECIPE about 55 % (non-twin noise is gone, the guard against the best part is tight), ADDS-UP about 15 % (inside ADOPT-RECIPE), NO about 45 %. The parts are expected NOT to add up (the predicted +0.002 is below the +0.003 'adds up' bar): the most likely outcome is the simplest passing single part.

## 1. Card text
- **Statement:** the gain parts that passed their own cards, combined (with S4 if it holds), keep the gain of the best of them on the seed-twin heads and stay within the champion's guards, at two training stages at most and an unchanged architecture.
- **Why the seed-twin head seeds 2 / 3:** e373_cb (the champion), e376_mx (max-positive alone), e379_d (distillation alone) and e372_s4 (S4 of cb) all use head seeds 2 / 3 on caches 0 / 1: training c on the same seeds makes the 2 x 2 (champion, MP, KD, MP + KD) and the S4 stage seed-twinned at no extra GPU cost; the twin is proven by the first batch's frames and the step-0 init hash (E379's checks) on the simplified code.
- **E2E / D55:** no part adds a rule step; the named hard steps stay (the top-K index choice, the allowed-pair gate on the inputs, the four output thresholds of the decode, the frozen pyramid).
- **Not claimed:** TEST; K 384; temporal fusion (E384, the next card); e2 (the lead's D106 decision); the speed path's training time (E383's verdict).
- **Inputs:** ready: `--pos-loss`, `--distill-from` (e3df44fb), jlive_train.py (S4), `tools/jarm.py`, `jtoken_eval.py choose-cell / score / ensemble / --live / timing`. Open: the verdicts of E372, E378, E379 by 10:30; the D117 / D119 simplified code (lands 07:00 - 08:30).
- **Launch check (before c starts, written into section 3):** on the simplified code at the launch commit, `training/grid/test_jtoken.py`'s champion bit-for-bit test is green and `jarm.py curve E385 c_s<S> --ref e373_cb_s<S>` (the early losses and the data-order fingerprint) agrees with E373 cb on the twin seeds (plus, with KD, the twin first loss of E379: 4.5520 against E373's 4.5516).
- **Schedule:** composition at 10:30 (section 3), launch check, start at the latest 11:30 on one slot (MP + KD: 2 x ~70 min, + read-outs; MP + S4: 2 x (35 + 30) min) so that the verdict can feed the final report at 15:30; a run that cannot start by 11:30 is moved to the night and the report says so.
- **Budget per composition:** MP + KD 2.5 GPU h (incl. dumps); MP + S4 2.2; no run 0.
- **Lens / group / sources / status:** D117 point 4 / D119; status v2 after the critic, the composition open until 10:30.
- **Referee:** v1 REVISE (fresh `ar-critic`, cards, ~02:30: 6 blocking, 12 should-fix, nits, all taken in above); v2: pending a second look; a third short look once the composition is filled.

## 2. Detail plan
| step | what exactly | where | done when |
|---|---|---|---|
| check | the max-head ensemble (CPU, now): decides the teachers if MP and KD are both taken | `temp/e385/maxens.sh` | number in section 3 |
| compose | at 10:30 fill the table from the verdicts; write the exact command lines; the re-measured KD bands if MP + KD | this card | committed before any run |
| critic | a short look on the filled composition | Fable | G2 yes |
| launch check | test_jtoken bit for bit; jarm curve vs e373_cb on the twin seeds | Fable | section 3 |
| train | stage 1 c_s0, c_s1 (head seeds 2 / 3); stage 2 if S4 | `jarm.py run E385 c_s<S>`; `run.py ... jlive_train.py` | done flags |
| read-out | the verdict cell vs the champion and vs the best part; live if S4; drop 0.17; paired timing | `jtoken_eval.py` | cards |
| decide | the verdict above | by hand | result section, ledger notes, commit, one line to the lead; the champion line through the lead (D<n>) |

## 3. Read-out log
(the max-head ensemble check, the composition at 10:30, the launch check, the cells: written here before the runs they govern)
