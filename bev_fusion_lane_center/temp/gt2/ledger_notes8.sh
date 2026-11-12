cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E314.p_s0 --result "snapped p>=0.1 topo F1 0.6852 (+0.0593 vs plain, +0.0027 vs E312), lanes found +0.1239, pair recall 0.353" --decision "TIE by hand: card rule (+0.01 over E312 not reached); E312 stays the final system" 2>&1 | tail -n 1
$L note E314.p_s1 --result "snapped p>=0.1 topo F1 0.6846 (+0.0597 vs plain, -0.0018 vs E312), lanes found +0.1347, pair recall 0.355" --decision "TIE by hand: card rule (+0.01 over E312 not reached); E312 stays the final system" 2>&1 | tail -n 1
