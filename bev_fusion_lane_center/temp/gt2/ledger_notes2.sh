cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E305.e_s0 --result "topo F1 0.6328 (+0.0069 vs plain, +0.0008 vs p2), lanes found +0.030, junction recall +0.069, pair R/P 0.206/0.654, precision at p2 recall +0.053" --decision "TIE by hand: card rule (+0.01 over p2 not reached), kept and refined" 2>&1 | tail -n 1
$L note E305.e_s1 --result "topo F1 0.6302 (+0.0053 vs plain, +0.0016 vs p2), lanes found +0.033, junction recall +0.095, pair R/P 0.218/0.662, precision at p2 recall +0.001" --decision "TIE by hand: card rule (+0.01 over p2 not reached), kept and refined" 2>&1 | tail -n 1
