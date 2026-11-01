# E244 Matching-free Poisson (PHD) set likelihood instead of the clamped hinge recall term

The card (read by `experiment.py` and `decide.py`; schema `autoresearch/schemas/experiment.schema.json`):

```json card
{
 "id": "E244",
 "card": "H19.F3-1",
 "title": "Poisson (PHD) set likelihood of the ground-truth lane points under the node-edge intensity replaces the clamped hinge recall term (4 u fine-tune of the fast variant j1 192 / 96)",
 "kind": "block",
 "deterministic": false,
 "metric": "Q",
 "direction": "higher",
 "band": 0.01,
 "prediction": {
  "value": 0.012,
  "vs": "c"
 },
 "variant": "a",
 "arms": {
  "a": {
   "recipe": "p16_train.py --design D7 --init e123_D7_r3_j1 --u 4 --lr 0.0005 --stem_lr 0.3 --warm_frac 0.4 --set <j1 config as _run_j1eq.sh> --w phd=0.3 rec=0 ; seeds 0 and 1; evaluated at tau 0.5, fork_prune 0.4",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0,
    1
   ],
   "planned_names": [
    "fast_phd",
    "fast_phd_s1"
   ],
   "note": "tau 0.5 declared before any run"
  },
  "b": {
   "recipe": "as a but the OLD hinge recall (rec=1) with the clamp lifted from 3 m to 8 m (env P19_REC_CLAMP=8), phd=0; isolates the dead zone from the form of the loss",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0
   ],
   "planned_names": [
    "fast_phd_clamp8"
   ]
  },
  "c": {
   "recipe": "existing equal-budget control: e18_D7_j1eq (seed 0) and e18_D7_j1eqs1 (seed 1), recall weight 1, same 4 u recipe; tau = better of 0.5 / 0.7 on VAL seed 0 (pre-declared)",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0,
    1
   ],
   "planned_names": [
    "fast_eq",
    "fast_eq_s1"
   ],
   "run_ids": [
    "E219.j1eq",
    "E219.j1eqs1"
   ]
  },
  "r": {
   "recipe": "existing reference j1rc / j1rcs1 (recall weight 3, registry fast_recall)",
   "base": "e123_D7_r3_j1",
   "seeds": [
    0,
    1
   ],
   "planned_names": [
    "fast_recall",
    "fast_recall_s1"
   ],
   "run_ids": [
    "E219.j1rc",
    "E219.j1rcs1"
   ]
  }
 },
 "control": "c",
 "rule": [
  {
   "metric": "Q",
   "min_effect": 0.01,
   "note": "headline, beyond the band against the equal-budget control"
  },
  {
   "metric": "lanes_found",
   "min_effect": 0.05
  },
  {
   "metric": "R@1_far",
   "min_effect": 0.03
  },
  {
   "metric": "topo_F1",
   "max_loss": 0.005
  },
  {
   "metric": "Q",
   "vs": "r",
   "min_effect": 0.003,
   "note": "better than the recall weight 3 fine-tune j1rc"
  },
  {
   "metric": "lanes_found",
   "vs": "r",
   "max_loss": 0.02
  }
 ],
 "seeds": [
  0,
  1
 ],
 "budget_gpu_h": 1.35,
 "kill": "toy test fails; < 15 % of the missed far points in the dead zone and arm b not the explanation; after 1 u (1200 steps) val250 lanes found more than 0.03 below j1 or loss not finite; Q of a below c - 0.010 after both seeds",
 "evaluate": {
  "split": "val",
  "compare": "c"
 }
}
```


One file per experiment; it grows from card (P2) to detail plan (P3 start, block experiments only) to summary (P3 end). Machine-readable part: `experiment.json` next to this file.

## 0. What the tools run (the fields of `experiment.json`)
| field | value |
|---|---|
| id / card / title | E244 / H19.F3-1 / Poisson set likelihood of the ground-truth lane points under the node-edge intensity replaces the clamped hinge recall term (fine-tune of the fast variant j1 192 / 96) |
| kind | block (new loss branch in `training/losses/p16_loss.py`, `graph_losses`, behind a weight key `phd`; default 0 = identical to today) |
| deterministic | false (training seed matters: seeds 0 and 1) |
| metric, direction, band | Q, higher, 0.010 (also reported: S, lanes_found, R@1_far, topo_F1, J, vert100, ms) |
| prediction | +0.012 on Q against arm c (plain fine-tune, recall weight 1); lanes_found +0.06, R@1_far +0.05, topo_F1 +0.010 |
| arms | a: PHD likelihood, 4 u fine-tune from `e123_D7_r3_j1`, seeds 0 and 1, tau 0.5; b: ablation, hinge recall with the clamp lifted from 3 m to 8 m (seed 0); c: equal-budget control j1eq / j1eqs1 (`e18_D7_j1eq`, `e18_D7_j1eqs1`, recall weight 1), tau = the better of 0.5 / 0.7 on VAL seed 0; r: reference j1rc / j1rcs1 (recall weight 3, `fast_recall`) |
| control | c |
| rule | Q min_effect 0.010 vs c; lanes_found min_effect 0.05 vs c; R@1_far min_effect 0.03 vs c; topo_F1 max_loss 0.005 vs c; Q min_effect 0.003 vs r; lanes_found max_loss 0.02 vs r |
| budget_gpu_h, kill | 1.35 (0.15 shared set-up, 2 x 0.4 arm a, 0.4 arm b), see section 1 |
| evaluate | split val (873 frames), compare c; new registry entries (see inputs) |

## 1. Card (written in P2, checked by ar-referee)
| | |
|---|---|
| lens | foundations (F3 formulation, F2 loss) |
| group | A3 sparse lane graph, A4 training regime (gap 1 coverage at range, gap 2 connectivity) |
| sources | deep/2203.07980_pmb-nll.md (sections 5.1 - 5.3), deep/2509.08104_apml.md (soft assignment, mass term), register K02 (survival / hazard, the path validity pi); own loss `training/losses/p16_loss.py` `graph_losses`; E219 (rc1, rc2, rc4, j1rc, j1eq, f1eq), E217, E233; new K id: K193 (see below) |
| status | proposed |

- **Statement** (falsifiable): Replacing the soft-minimum recall term of `graph_losses` (smooth-L1 of min over edges of [distance + 2 m x (1 - pi)], clamped at 3 m) by the matching-free Poisson set log-likelihood of the ground-truth lane points under the intensity of the valid edges, with a mass term sum(pi x length), gives a 4 u fine-tune of the fast variant j1 192 / 96 with Q at least 0.010 above the equal-budget plain fine-tune j1eq (Q 0.735 / 0.731, mean 0.733) and lanes found at least 0.05 above it (j1eq 0.674 / 0.677), and with Q at least 0.003 above the recall-weight-3 fine-tune j1rc (Q 0.739 / 0.743), which is the one thing the recall weight never delivered.
- **Mechanism** (why it should work):
  1. The present recall term has a gradient dead zone. In `graph_losses` the soft minimum `dg_ = min_i(distance(g, edge_i) + 2 (1 - pi_i))` is clamped at 3 m before the smooth-L1: a ground-truth point whose best edge has distance + price >= 3 m (for example a hypothesis 1.2 m off with validity 0.1, or no hypothesis within 3 m) contributes no gradient to anything. The weight of the term (rc1 3, rc2 6, rc4 range weight) scales the points that already have a gradient; it cannot reach the points in the dead zone. That fits the record: weight 6 gives no further Q (E219), E217 shows that all valid nodes cover 85 % of the far points but the admitted ones 74 %, and E233 shows the recall weight lifts every bin but leaves 0.08 to the control beyond 40 m.
  2. The Poisson log-likelihood `L = [ rho sum_i pi_i l_i - sum_j log(lambda(g_j) + eps) ] / n_gt`, `lambda(x) = rho sum_i pi_i k(d_i(x))`, k = 0.8 Gaussian (sigma 0.6 m) + 0.2 Cauchy (gamma 2 m) cross-track density, d_i the distance to the edge segment (already computed as `sd` in `graph_losses`), rho = 1 point per m (the loss uses every 2nd 0.5 m point), eps = 1e-3, gives a gradient to every edge within about 10 m of a point, because the log of a sum is scale free (the relative share k_ij / lambda_j does not vanish with distance as long as the heavy tail is above eps). The per-edge optimum of pi is the supported fraction of its length (derivation: d/dpi_i = rho l_i - rho sum_j k_ij / lambda_j = 0), so pi becomes a coverage-calibrated score and a duplicate pair splits its mass instead of both going to 1. That is what the PMB paper's Poisson part provides (a home for undetected points, section 5.1 of the deep note) in the matching-free PHD form; the hard assignment of the paper's training loss is not used.
  3. pi is the path validity (pi_mode min), so a covered point pulls up the whole chain of ancestors: coverage that is connected to the tree is rewarded (the recall variants gain lanes found 0.06 - 0.07 but topo_R only 0.01: topo_R / lanes found falls from 0.98 (champion) to 0.92 (fast_recall) and 0.89 (recall), i.e. the added coverage is partly disconnected; this is a reading of the stored cards, not yet a result).
  4. Because pi becomes calibrated, the admission threshold has a meaning: the arm is judged at tau 0.5 (declared here, before any run); the control gets both 0.5 and 0.7 and the better one on VAL seed 0 (a fair choice for the control).
- **Prediction** (VAL 873 frames, `evaluate.py`, fork pruning 0.4 as the recall variants, against arm c = j1eq / j1eqs1, mean of seeds): Q +0.012 (0.745), lanes_found +0.06 (0.735), R@1_far +0.05 (0.74 from 0.69), topo_F1 +0.010, vert100 within +-0.5 of 14.9, ms per frame unchanged (the loss is training only). Against arm r (j1rc): Q +0.005, lanes_found within -0.02. Arm b (clamp 8 m): lanes_found between c and a (the dead zone alone explains part, not all); if b equals a, the form of the loss is not the lever and the cheaper b is the finding.
- **Decision rule** (noise band 0.010 on Q, 2 seeds paired by seed, paired scene bootstrap interval from `evaluate.py --compare`): KEEP if all clauses of section 0 hold with both seeds on the same side and the interval of Q excludes 0. Inside the band (|dQ| <= 0.010): TIE, the plain fine-tune c is kept (no new code on the training path). A Q gain that holds only against c but misses the clause against r: report as "form of the loss" result, no promotion. INCONCLUSIVE (one seed beyond the band, or seeds disagree): add seed 2 for a and c only (0.8 GPU h), nothing else. Tau multiplicity: only the declared tau 0.5 for a; the choice of c's tau is made on seed 0 before a is evaluated.
- **Test design**: arms as in section 0. Controls: null = j1 itself (`fast`: Q 0.728, lanes found 0.671, R@1_far 0.677) and the equal-budget control c (same 4 u recipe, recall weight 1, same `--warm_frac 0.4`, lr 5e-4, stem lr 0.3, M 192, K0 96, seeds 0 / 1; existing runs, reused, not re-trained); matched reference r on the same frames; seed twin = seeds 0 and 1 of every arm; shuffled-GT sanity (layer 2): the loss evaluated with ground-truth points shifted across frames must be higher than the true pairing by at least 1 nat per point on 200 TRAIN frames (a loss that does not see the geometry fails). Layer 0 before any training: toy cases with a known answer (one edge of length 10 m, pi = 1, ten points on it: L = rho l / n - mean log(lambda) with the closed form of the Gaussian mixture at its centre; two identical edges: optimum of pi_1 + pi_2 is 1; no edge: L = -log(eps) per point = 6.91; gradient is non-zero for a point 6 m from a pi = 0.1 edge). Layer 1 diagnostic (kills the card cheaply): on 200 TRAIN frames and the 873 VAL frames of the cached j1 raw nodes (`raw_val873_*.pkl`), the share of ground-truth points beyond 40 m that sit in the dead zone of the present term (clamp active, zero gradient), split into covered (a valid edge within 1 m) and missed points.
- **Cost** (GPU h / coding days / risk): 1.35 GPU h on the RTX 3060 (set-up 0.15: registry entries and cards for the four existing control runs j1eq, j1eqs1, j1rc, j1rcs1 at tau 0.5 / 0.7, shared with E245 and E247; 2 x 0.4 for a; 0.4 for b; one job at a time or two jobs max, GPU < 9 GB as the loop-18 runs); coding 0.5 day (about 40 lines in `graph_losses` re-using the tensor `sd`, one weight key, one spec test, one registry entry per arm); risk medium: the likelihood calibrates pi instead of inflating it, so lanes found at tau 0.7 may not move (that is why tau 0.5 is declared) and the weight of the term (starting value 0.3, so that the gradient norm matches the recall term at weight 1) may be off by a factor 3; a shared-code change needs an `ar-reviewer` pass (mode code) and `evaluate.py --golden check` must still pass with the default weight 0.
- **Kill criterion**: (1) a layer-0 toy test fails; (2) the dead-zone diagnostic finds < 15 % of the missed far points in the dead zone AND b is not planned as the explanation (then mechanism 1 is wrong and mechanism 2 alone is a smaller card); (3) after 1 u (1200 steps) the val250 lanes found is more than 0.03 below the start j1 or the loss is not finite (BN derail guard of the trainer fires); (4) after both seeds Q of a is below c - 0.010. Any of them stops the remaining runs of the card.
- **Differentiable path**: yes. Graph-based loss (vertices and edges of the tree), dense tensors [B, G, Nn] as in the present recall term, a log of a sum instead of a minimum; no argmax, no matching, no rule step (the selection steps inside the builder are the existing ones); the gradient reaches pi (hence survival, child validity, ray verification) and the node positions. Nothing changes at inference (ms per frame equal to j1rc: 20 - 22 ms).
- **Inputs**: ready: `e123_D7_r3_j1` init, the training launcher pattern of `experiments/E219_recall-weight-finetune-under-pruning/code/_run_j1eq.sh` (copy with `--w phd=0.3 rec=0`), cached raw nodes of VAL for j1 / j1rc, the control runs and their weights. To build: loss branch (`graph_losses`, key `phd`, optional env `P19_REC_CLAMP` for arm b), spec test file, registry entries `fast_eq`, `fast_eq_s1`, `fast_recall_s1`, `fast_phd`, `fast_phd_s1`, `fast_phd_clamp8` (not champion / control entries, so unprotected), cards of the old runs through `evaluate.py` (their loop-18 files in `runs/e18/rc_eval/` lack topo_F1 and R@1_far). New technique: **K193** "matching-free Poisson / PHD set likelihood of ground-truth lane points under the edge intensity (Bernoulli-free PMB-NLL form)", source 2203.07980 (idea only), must be added to `literature/register.md` before the first run.
- **Referee** (2026-10-25, ar-referee): REVISE (the best-aimed card of the set at known gap 1; run tonight after E240, staged). Verified in `p16_loss.py`: the recall term is `sd.min(-1).clamp(max=3.0)`, so the dead zone exists; arm b (clamp 8 m, one env variable, no new loss code) tests that mechanism directly and is the cheap first step. Revise: (1) STAGE it: diagnostic (CPU) -> arm b seed 0 -> only then the new loss (a) seeds 0 / 1; a null of `a` with one hand-guessed weight (0.3, "may be off by 3") says little about the idea, so allow one weight change on val250 at 1 u before declaring it dead; (2) the closed question "larger recall weights" can be re-opened only with a different far-range mechanism: write the dead-zone argument explicitly as the re-open condition (it differs from a weight, but it is not one of the two examples named there); (3) tool arithmetic: decide.py raises a Q clause on the headline metric to the band, so "Q vs r min_effect 0.003" becomes +0.010 over j1rc; add `"band": 0.003` to that clause, otherwise KEEP is nearly unreachable (Q effects of all earlier fine-tunes are <= 0.008); (4) predicted Q +0.012 and lanes found +0.06 lie outside the class of E219 results; shrink Q to about +0.006 or name the mechanism that beats weight 3 (the card does: tail points, so keep lanes_found / R@1_far as the main claim); (5) compare at matched vertices per 100 m (E236 tau helper), not at tau 0.5 against the control's tau; a at tau 0.5 is declared, good, but Q at a different vertex count is the confound E236 exists to expose; (6) `rec=0` with `phd=0.3` removes the whole hinge, not only its clamp: add arm r' (rec=1 + phd) only if time; (7) the shared `p16_loss.py` edit must be default-off and land before the next launch (a concurrent edit already broke E230.j1p once); K193 into the register first. Cost: solo 14 min per fine-tune, 0.7 GPU h for b + a x 2 plus 0.5 day coding with reviewer: about 2.5 h wall including coding.

## 2. Detail plan (block experiments only; written by ar-planner when the loop reaches the card)
What would make it fail, in our own words: the likelihood is satisfied by calibrating pi downwards (mass term) so that fewer edges pass tau; the heavy-tailed kernel attracts far edges to the wrong lane; the weight is mis-scaled.

| step | what exactly | where (file / function) | done when | budget |
|---|---|---|---|---|
| code | `phd_recall(res, b)` from the tensor `sd` of `graph_losses`; key `phd` in DEFAULT_W | `training/losses/p16_loss.py` | spec test passes (toy cases of the card); gradient reaches `pi` and `pos` | 0.5 day |
| diagnostic | dead-zone share per range bin on TRAIN 200 + VAL cached raw | `temp/loop19/` one-off script | numbers in the experiment file | 0.05 GPU h |
| train | arms a (seeds 0, 1), b (seed 0), launcher as `_run_j1eq.sh` with `--w phd=0.3 rec=0` | `run.py --id E244.a0 ...` | done flag, metrics | 1.2 GPU h |
| evaluate | registry names above, VAL, `--compare fast_eq` | `evaluate.py` | metric cards | 0.15 GPU h |
| decide | the rule above | `decide.py --experiment E244 --write` | verdict in the ledger | - |

## 3. Summary (2026-10-26, runs E244.b0, E244.a0 (cut); VAL 873, fork pruning 0.4, `evaluate.py`; diagnostics `diag_dead_zone.txt`)
- **Decision: arm b CUT as a lever (clamp 8 m = j1eq); arm a not decided, partial reading negative; INCONCLUSIVE by design (one seed, half schedule)**. No `decide.py` verdict was written: the rule needs two finished seeds of arm a.
- **Step 1 (dead zone)**: the gradient dead zone (distance to the nearest predicted node >= 3 m) holds 6.4-7.6 % of the far (> 40 m) ground-truth points and 30.4-33.5 % of the missed far points; the kill threshold (15 % and arm b not planned) was not met, so the staged plan went on.
- **Arm b (clamp 8 m, recall weight 1, seed 0)**: Q 0.733, lanes found 0.678, R@1 far 0.691, topo_F1 0.738, vert100 14.77, equal to the plain fine-tune j1eq (0.735 / 0.674 / 0.688 stored seed 0; single seed, differences -0.0026 / +0.0034 / +0.0027 inside the seed SDs): the clamp is not the lever. The prediction "lanes found between j1eq and j1rc" missed.
- **Arm a (Poisson set likelihood, `phd` 0.65 from the gradient calibration, `rec` 0, tau 0.5 declared, seed 0)**: the run was cut at iteration 3200 of 4800 (E244.a0 ledger note): other chats' jobs held the machine at load average 30-64 and a second GPU process ran for 100 minutes, so 100 iterations took 1400-2400 s instead of 60-110 s. Only the 50 % checkpoint exists (`e24_j1phd_s0_at50`). Logged losses: the Poisson term falls from 2.28 to 1.56 by iteration 300 and then stays at 1.53-1.55; the hinge recall term it replaced rose from 0.27 to 1.05 (the plain fine-tune ends at 0.31), so the plateau came early and the replaced term got worse; the loss is not "healthy" evidence.
- **Partial reading, 50 % checkpoint, same point of the schedule as the E241 checkpoints of seed 2 (other seed)**: at the declared tau 0.5: Q 0.634, lanes found 0.392, vert100 21.2 (far outside the 14.3 budget). tau sweep on VAL (tuning on VAL, in favour of arm a): the best Q is at tau 0.12-0.2 (0.726-0.728) with 18.8-19.3 vertices per 100 m; the vertex radius `vtx_r` 3.5 brings it to the budget (vert100 15.1) at Q 0.673, lanes found 0.613, R@1 far 0.627. Against j1eq at 50 % (Q 0.730, lanes found 0.671, R@1 far 0.676, vert100 14.6) and j1rc at 50 % (0.736 / 0.746 / 0.740 / 15.0): arm a is behind by 0.057 Q at the matched budget, lanes found -0.06. The node score of the Poisson loss is badly calibrated against tau (the declared 0.5 was a guess) and the graph is vertex-hungry.
- **What it says**: as implemented (set likelihood replaces the recall term, nothing else) the Poisson loss does not lift lanes found or Q; the 50 % reading is far enough below the equal-budget control that two finished seeds are unlikely to reverse it, but that is a judgement, not a measurement. A fair version needs the declared tau from a calibration (not 0.5), the dup/precision terms re-weighted against the new node scores, and finished seeds on a quiet machine.
- **Weaknesses**: one seed, half schedule (the learning-rate schedule is not annealed at 50 %; the E241 checkpoints are in the same state), checkpoint comparison against another seed, tau and `vtx_r` tuned on VAL for arm a only.
- **Next**: re-run a0 / a1 on a quiet machine with a tau calibration step (K193 stays in the register as "tried, not lifting, partial"); layers 0-1 done (toy tests 7/7 and gradient calibration 0.655 were console outputs and were not stored).
