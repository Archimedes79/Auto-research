#!/bin/bash
cd "$(dirname "$0")/../.."
L="python3 ../autoresearch/tools/ledger.py"
$L note E379.time_e2b --result "QUIET, batch 1, bf16: champion path 16.21 ms (B' 11.20 + head), e2 two heads 21.84 ms in sequence / 25.19 ms stacked (+5.63 / +8.98 ms, 2,450,274 vs 2,323,143 parameters); the E324 s yardstick of the gate 19.7 ms" --decision "by hand: the E379 cost gate (D106, the card's written rule): MISS, e2 above 19.7 ms on both paths; the lead closed E379 at 05:35 (d NO at the recipe cell / TIE in substance, e2 MISS on cost, d4 exploratory); card section 4"
$L note E379.time_e2 --result "first run, flagged NOT QUIET only by the tool's after-sample artefact (no 2.5 s wait): whole path 16.29 / 22.02 / 25.30 ms, the QUIET repeat E379.time_e2b is the one cited"
