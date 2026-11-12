p = "temp/e287/start_pairs.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('case $ARM in p0) FLAGS="--no-order --no-evidence --cross-w 0";; p1) FLAGS="--no-evidence";; p2) FLAGS="";; *) echo "arm?"; exit 2;; esac',
              'case $ARM in p0) FLAGS="--no-order --no-evidence --cross-w 0";; p1) FLAGS="--no-evidence";; p2|q2) FLAGS="";; *) echo "arm?"; exit 2;; esac')
open(p, "w", encoding="utf-8").write(s)
c = "experiments/E287_junction-pair-head-v2/experiment.md"
t = open(c, encoding="utf-8").read()
old = '  "p2": {"recipe": "+ the evidence along the Hermite curve (lane-centre probability of B, zone, conflict core, stop line, crossing): the full head"}\n }'
new = '  "p2": {"recipe": "+ the evidence along the Hermite curve (lane-centre probability of B, zone, conflict core, stop line, crossing): the full head"},\n  "q2": {"recipe": "the full head of p2 on the maps of the JOINT arm jj of E286 (the pyramid trained together with the junction head); the delivered graph under the connectors is still B\'s own"}\n }'
assert old in t
t = t.replace(old, new, 1)
open(c, "w", encoding="utf-8").write(t)
print("ok")
