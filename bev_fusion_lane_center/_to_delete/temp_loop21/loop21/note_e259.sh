#!/bin/bash
cd "$(dirname "$0")/../.."
L=../autoresearch/tools/ledger.py
python3 $L note E259.a0 --result "clean-up off on 7 models: dQ +0.0001 .. +0.0027, dvert100 +0.44 .. +0.73, dFL +0.002 .. +0.003, dlanes_found +0.017 .. +0.021, dtopo_F1 -0.005 .. -0.006; wall timing unreadable (load 4 - 15)" --decision "the clean-up stays (vert100 guard fails on all seven models); no free lunch at matched vertices"
python3 $L note E259.b0 --result "CPU time under load 9: anchoring off -4.9 / -4.3 ms (10 of 10 reps), relocation off -5.4 / -3.4 (7 of 10), clean-up off not readable, dp stage costs ~+0.7 - 1.5 ms" --decision "stage ranking consistent with E258; timing provisional; process CPU time does not remove the load effect (N18)"
