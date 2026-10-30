#!/bin/bash
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/ledger.py note E257.a0 --notes "D20 adopted 2026-10-30: simplify_dp / assemble(dp_eps), registry fast_recall_dp, champion_dp reproduced by evaluate.py (Q 0.739 / 0.735, vert100 9.86 / 9.52); code review passed after fixes"
