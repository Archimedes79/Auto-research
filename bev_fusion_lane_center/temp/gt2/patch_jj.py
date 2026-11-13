p = "temp/e286/start_jmaps.sh"
s = open(p, encoding="utf-8").read()
s = s.replace('case $ARM in j0) CTX=none;; j1) CTX=dil;; j2) CTX=attn;; *) echo "arm?"; exit 2;; esac', 'EXTRA=""\ncase $ARM in j0) CTX=none;; j1) CTX=dil;; j2) CTX=attn;; jj) CTX=${JJ_CTX:-dil}; EXTRA="--joint";; *) echo "arm?"; exit 2;; esac')
s = s.replace("--ctx $CTX --smoke --workers 3", "--ctx $CTX $EXTRA --smoke --workers 3")
s = s.replace("--ctx $CTX --steps 3000 --workers 3", "--ctx $CTX $EXTRA --steps 3000 --workers 3")
open(p, "w", encoding="utf-8").write(s)
c = "experiments/E286_junction-pixel-maps/experiment.md"
t = open(c, encoding="utf-8").read()
old = '  "j2": {"recipe": "`jmaps_train.py --ctx attn` (+ one self-attention layer over 25 x 25 tokens at 4 m, learned position)"}\n }'
new = '  "j2": {"recipe": "`jmaps_train.py --ctx attn` (+ one self-attention layer over 25 x 25 tokens at 4 m, learned position)"},\n  "jj": {"recipe": "`jmaps_train.py --joint` with the context depth of the best of j0 / j1 / j2 (the pyramid of B trained together with the head, B\'s own pixel loss kept so that the maps of B do not drift; judged on the maps only, B\'s nodes / links no longer match such a pixel stage)"}\n }'
assert old in t
t = t.replace(old, new, 1)
open(c, "w", encoding="utf-8").write(t)
print("ok")
