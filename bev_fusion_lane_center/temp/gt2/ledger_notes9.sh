cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E298.w_s0 --result "driven pair recall 0.495 vs 0.373 (E312), overall pair recall 0.331 / precision 0.670; snapped p>=0.1 topo F1 0.6766 (-0.0059 vs E312)" --decision "KEEP by hand: card rule (driven pair recall +0.03 on both seeds), guards held" 2>&1 | tail -n 1
$L note E298.w_s1 --result "driven pair recall 0.511 vs 0.398 (E312), overall pair recall 0.346 / precision 0.647; snapped p>=0.1 topo F1 0.6865 (+0.0001 vs E312)" --decision "KEEP by hand: card rule (driven pair recall +0.03 on both seeds), guards held" 2>&1 | tail -n 1
