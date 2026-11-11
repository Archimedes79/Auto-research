# E295 A position spread per node (brief: "Eine Positionsunsicherheit zu") (D82)

Card of 2026-11-11 (plan card, after E289 step 1b). Literature of the day (`scan_2026-11-11_pruning_uncertainty_time.md`, idea 3): a Gaussian log-likelihood with a per-node sigma (beta-NLL, Seitzer et al. 2022), the KL-loss variance vote (He et al. 2019: the read-out position is the 1/sigma^2-weighted mean of the neighbours' votes), calibration by ENCE (Levi et al.). The sigma is used downstream as a weight of the matching, of the pruning and of the time fusion. Note from E289: a large part of the node jitter is the GRID PHASE along the line (a vertex position along the line is arbitrary), the line itself is stable; the sigma along the line is therefore large by nature and across the line small: a 2 x 2 covariance (or sigma along / across the heading) is the right output.

## 0. The card (what the tools run)
```json card
{
 "id": "E295",
 "card": "",
 "title": "a spread (sigma along and across the heading) for every node of B, trained with beta-NLL on the frozen node head (a side stage like E284), and the sigma-weighted read-out position, against the plain offset position",
 "kind": "block",
 "deterministic": false,
 "metric": "ence",
 "direction": "lower",
 "band": 0.05,
 "prediction": {"value": 0.15, "vs": "c"},
 "arms": {
  "c": {"recipe": "control = B's node position (the offset head), the same sigma for every node (the global RMS error of the node position, 0.28 m): ENCE of the constant sigma is the baseline of the calibration"},
  "s": {"recipe": "a head of 0.1 M parameters on the node tokens: sigma along / across (softplus), beta-NLL (beta 0.5) on the offset error to the GT node of the cell, B frozen, 2000 steps, 2 seeds; read-out: the 1/sigma^2-weighted mean of the node and its 8 neighbours' votes"}
 },
 "control": "c",
 "variant": "s",
 "pairing": "position",
 "seeds": [0, 1],
 "rule": [
  {"metric": "ence", "vs": "c", "max_loss": 0.0},
  {"metric": "pos_err_median", "vs": "c", "max_loss": 0.02},
  {"metric": "F1@1.0", "vs": "c", "max_loss": 0.01}
 ],
 "budget_gpu_h": 0.5,
 "kill": "the sigma does not correlate with the error (Spearman below 0.2 on 20,000 nodes): the head has learned a constant",
 "evaluate": {"split": "val"}
}
```
By hand (label "by hand: calibration"): the headline is the calibration, ENCE (expected normalised calibration error) of sigma against the node position error on VAL nodes, lower is better; the guards are the position error (median, the card of B: 0.28 m) and F1@1.0 of the delivered graph with the sigma-weighted position. KEEP when ENCE is lower than the constant-sigma baseline by the band (0.05) on both seeds with the guards held; TIE kept and refined (D82): then the sigma goes into the matching of E287 and the pruning (nodes with a large sigma across the line are suppressed first). Reported: Spearman(sigma, error), sigma along vs across, the share of nodes with sigma across > 0.5 m (the junction nodes).
