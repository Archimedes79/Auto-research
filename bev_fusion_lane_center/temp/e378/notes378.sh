#!/bin/bash
# E378 ledger notes after the verdict (by hand, the card's rule v2)
cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py note"
$L E378.x_s1 --result "trained 12000 steps (slot 1, 03:28-04:15, Fable), pinned cell 0.3/0.2/0.5/off" --decision "NO at (i), c not run (by hand: the card's rule v2)" --notes "recipe cell (mean of both TRAIN tables) 0.3/0.2/0.5/off = x_s1's own cell"
$L E378.read_x_s1 --result "VAL 873 at the recipe cell 0.3/0.2/0.5/off vs e373_cb_s1 at D115: topo F1 +0.0000 [-0.0046,+0.0051], junction +0.0123, lanes found -0.0106 [-0.0177,-0.0029]" --decision "NO at (i)"
$L E378.recipe_x_s0 --result "VAL 873 at the recipe cell 0.3/0.2/0.5/off vs e373_cb_s0 at D115: topo F1 +0.0030 [-0.0023,+0.0108], junction +0.0379 [+0.0212,+0.0630], lanes found -0.0015" --decision "NO at (i)"
$L E378.d115_x_s1 --result "reported: x_s1 at D115 vs e373_cb_s1: topo F1 +0.0015 [-0.0031,+0.0065], junction -0.0016, lanes +0.0011" --decision "reported only"
echo notes done
