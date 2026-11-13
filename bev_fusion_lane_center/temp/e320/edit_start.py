p = "temp/e320/start_pairs.sh"
s = open(p, encoding="utf-8").read()
i = s.index("  --hyp ")
new_tail = '''  --hyp "The snap of the read-out (a rule on the output) carries most of the topology gain of E312: two thirds of it is the wiring (one nearest piece per end), a third the end geometry; a learned soft attachment of the connector ends to the nodes of B's head, trained by the geometry loss on the final curve and a cross entropy over the candidates, replaces the end geometry inside the network (differentiable)." \
  --pred "topo F1 of the primary read-out (attached ends + one nearest piece per end) is 0.006 below E312 snapped, 0.0059 / 0.0044 above E312 snap-nogeom (u'), at the KEEP bound" \
  --rule "E320 card section 0 (lane lead's final form): by hand (jends_eval.py score --attach --snap --snap-nogeom --thr 0.1, differences to the plain graph); KEEP when on both seeds topo F1 of arm a >= E312 snap-nogeom + 0.005 (0.6756 / 0.6810, implies >= snapped - 0.01) with the guards held; TIE when a is in [snap-nogeom, snap-nogeom + 0.005) on a seed; CUT below snap-nogeom on either seed or a guard broken; secondary and tight read-outs reported, not in the rule; VAL 873, no TEST" \
  -- bash tools/env/_pyr.sh jends_train.py --name e320_a_s$S --seed $S --maps $MAPS --e2e --scales --maps-w 1.0 --steps 3000 --geom-w 0.5 --attach --workers 3
'''
open(p, "w", encoding="utf-8").write(s[:i] + new_tail)
