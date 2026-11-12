# E322 - soft wiring: the pieces a connector joins chosen by the network, not by a distance rule (stub, no run)

Stub of 2026-11-13 00:10, written by the lane chat (fork A) on the lane lead's order, so that the roadmap can point to it. No card block, no run, no number is claimed below beyond what is measured on E312.

## Question
After E320 (the connector ENDS are attached to nodes of B by a softmax): can the WIRING, the choice of which delivered piece ends a connector is joined to, be learned and soft instead of the rule of `qends_eval.add_connectors` (every piece end within 4 m / 45 degrees, or the one nearest piece per end of the snap)?

## Why it is the next step (measured on E312, VAL, p >= 0.1, seed 0, lane lead 2026-11-12)
The gap between the snapped (0.6825 topo F1) and the unsnapped (0.6438) read-out of E312 is 0.0387; of it 0.0268 (69 %) is the WIRING (one nearest piece per end instead of every piece in reach: `--snap --snap-nogeom` 0.6706) and 0.0119 (31 %) the geometry of the ends. E320 attacks the geometry (and reads out with the same one-piece wiring as the primary read-out); the wiring stays a named hard step there (the argmin over the delivered piece ends), and E320's tight-radius read-outs (1.5 m / 1.0 m) test whether a contact wiring (a threshold around the attached end) can replace the choice among several pieces. That is still a distance rule on the output, not a learned wiring: if a tight read-out reaches E312 snapped minus 0.01 (with the guards) a contact threshold suffices at the read-out, if not the one-piece choice is needed; either way a learned, soft wiring is what this card is about.

## Hypothesis (to be tested, not claimed)
A learned attachment of every connector end to the DELIVERED piece ends in reach (the line ends and starts of the delivered graph, which the forward path of the lane head has: soft section scores, link probabilities), with a dustbin for 'no piece', and a loss that sees the topology (the reachability of the GT lane graph through the connector, not only the distance of the ends), replaces the distance rule of the read-out without loss: topo F1 of the soft read-out at or above E312 snapped minus 0.01 on both seeds. The cost to check first: the delivered graph is a hard read-out of B's node / link probabilities (thresholds, the close-node step), so the candidates are either its hard line ends (an index choice, gradient only through the scores) or a soft version of the graph (E277 / E288 are the precedents: the soft step was a CUT, Matrix-NMS a TIE).

## Not decided here
Design, arms, rule, seeds and budget are written in the card of the run, before it, and only after E320 is measured (the card of E320: `experiments/E320_soft-attach-replaces-snap/experiment.md`).
