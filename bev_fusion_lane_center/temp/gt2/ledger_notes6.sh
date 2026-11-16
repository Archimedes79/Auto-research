cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E312.p_s0 --result "ends 0.745/0.771, pair R/P 0.300/0.674; snapped p>=0.1 topo F1 0.6825 (+0.0566 vs plain, +0.0262 vs E305), lanes found +0.1109" --decision "KEEP by hand: card rule (topo F1 over E305 > +0.01 on both seeds, snapped p>=0.1), guards held" 2>&1 | tail -n 1
$L note E312.p_s1 --result "ends 0.752/0.767, pair R/P 0.333/0.665; snapped p>=0.1 topo F1 0.6864 (+0.0615 vs plain, +0.0330 vs E305), lanes found +0.1286" --decision "KEEP by hand: card rule (topo F1 over E305 > +0.01 on both seeds, snapped p>=0.1), guards held" 2>&1 | tail -n 1
