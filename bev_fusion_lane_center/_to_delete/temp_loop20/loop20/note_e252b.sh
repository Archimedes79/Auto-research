#!/bin/bash
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/ledger.py note E252.b0 --result "referee follow-up (champion): permutation null of the Q gap 0.0524 has SD 0.0274, p two-sided 0.056; shift of the GT 0.3 / 0.5 / 1.0 m costs Q 0.0225 / 0.0698 / 0.2147; per-polyline median offset 0.157 m Boston vs 0.221 m Singapore; no GT within 1.5 m: 0.026 vs 0.056" --decision "E252 verdict unchanged: cap not supported (Boston better on every count)"
