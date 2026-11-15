#!/bin/bash
# scratch: test of the gate() function of chain.sh by sourcing it (the main code does not run when sourced) with stored maps runs as stand-ins; seed 9 only names the log kill_s9.log
cd "$(dirname "$0")/../.."
. temp/e319/chain.sh
for t in "e315_m_s0 e307_n_s0:0" "e313_s_s0 e307_n_s0:0" "e312_p_s0_maps e307_n_s0:1" "e320_a_s0_maps e307_n_s0:1" "nosuchrun e307_n_s0:2"; do
  GATE_RUNS="${t%:*}"; gate 9; G=$?
  echo "gate with '${t%:*}' -> $G (expected ${t##*:})"
done
rm -f temp/e319/kill_s9.log
