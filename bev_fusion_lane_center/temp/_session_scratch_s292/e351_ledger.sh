#!/bin/bash
# E351 ledger notes (results of the equality and timing runs; numbers from runs/eval/jgraph and runs/eval/jtiming)
cd /mnt/c/Auto-research/bev_fusion_lane_center || exit 1
L="python3 ../autoresearch/tools/ledger.py note"
$L E351.smoke_g1 --result "PARTIAL (8 of 873 frames): null and graph identical, 176 tensors" --decision "smoke, no verdict" --notes "jgraph.py check --max 8 --dump into a scratch runs dir; the run was a smoke of the capture and the dump path"
for R in e312_p_s0:19206 e312_p_s1:19206 e324_s_s0:24444 e324_s_s1:24444; do
  r=${R%%:*}; n=${R##*:}
  $L E351.eq_$r --result "TIER 1: graph forward equal to the eager forward (torch.equal and the same bits) on all 873 VAL frames, $n tensors; NULL control 0 differences" --decision "tier 1 by the verdict of jgraph.py check (by hand: replay, E351 card section 3)" --notes "runs/eval/jgraph/e351_equal_$r.json; dumps runs/e351_eager_$r and runs/e351_g1_$r; the cards scored from the dumps are not done (card section 3)"
done
$L E351.g1 --result "E312 c 14.733 g1 12.763 ms, -1.808 [-1.936, -1.681] = -12.3 %; E324 s c 19.468 g1 17.344, -1.997 [-2.408, -1.604] = -10.3 %; host load 0.87 / 0.88" --decision "by hand: the tool wrote NOT QUIET / INCONCLUSIVE because its GPU after-sample read the process's own tail (util 42 %, own context 1442 MiB); load was quiet; repeated as E351.g1b-d" --notes "runs/eval/jtiming/e351_graph_20261115.json; flaw fixed in jtiming.py (2.5 s pause after the synchronisation, uncommitted at the time of the note)"
$L E351.g1b --result "E312 c 14.730 g1 13.338, -1.711 [-2.097, -1.322] = -11.6 % (tool: TIE); E324 s c 19.929 g1 17.089, -3.039 [-3.525, -2.630] = -15.3 % (tool: ADOPT, bar 2.99 ms); load 0.81 / 0.75, GPU idle" --decision "by hand: one of four runs; see E351.g1d for the verdict" --notes "runs/eval/jtiming/e351_graph_20261115b.json; repeat after the sampling fix"
$L E351.g1c --result "E312 -1.919 [-2.149, -1.739] = -13.1 %; E324 s -2.402 [-2.687, -2.090] = -12.3 %; load 0.82 / 0.78" --decision "by hand: pooled repeats (rule written 2026-11-15 11:33 after the first two runs: the median of the per-run drops against 15 % of the control median)" --notes "runs/eval/jtiming/e351_graph_20261115c.json"
$L E351.g1d --result "E312 -1.831 [-2.092, -1.550] = -12.5 %; E324 s -2.634 [-2.811, -2.475] = -13.4 %; load 1.20 / 1.06" --decision "TIE for both systems by the median of four quiet runs (E312 -12.4 %, E324 s -12.8 %, about -2.5 ms; one E324 s run reached 15 %): by hand: pooled repeats; eager stays by the rule, adoption is a judgement for the lead / Mirko" --notes "runs/eval/jtiming/e351_graph_20261115d.json; prediction -4 / -9 ms missed (the probe was measured at load 5 - 6); card section 3"
echo "ledger notes done"
