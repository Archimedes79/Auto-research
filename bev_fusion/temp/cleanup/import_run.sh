#!/bin/bash
cd /mnt/c/Auto-research/autoresearch || exit 1
for P in "" /mnt/c/Auto-research/bev_fusion; do
  echo "== AR_PROJECT='$P'"
  AR_PROJECT=$P python3 - <<'EOF'
import importlib.util as u, os, sys
s = u.spec_from_file_location("ar_run", "tools/run.py")
m = u.module_from_spec(s)
try:
    s.loader.exec_module(m)
    print("imported; has", [n for n in ("state_of", "record", "command_alive", "twin_refusal", "is_holdout", "names_split", "ledger_row", "run_folder", "HOLD", "STALE_S", "test_log_refusal", "decided_ids") if hasattr(m, n)])
    print("P.root:", getattr(m, "P", None) and getattr(m.P, "root", None))
except BaseException as e:
    print("import failed:", type(e).__name__, str(e)[:160])
EOF
done
grep -n -E "^(P|HOLD|STALE_S|cfg) *=|^import project|^from project|import hold" tools/run.py | head -8
grep -n -E "^def (test_log_refusal|decided_ids|run_folder|test_log_opens)" tools/run.py