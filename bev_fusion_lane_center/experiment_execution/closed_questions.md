# Closed questions (lcr)

A question is closed when it was answered the same way twice or Mirko closed it. A hypothesis card that re-opens one must name a new mechanism (`../../autoresearch/METHOD.md`, P2).

| question | answer | evidence (experiment ids) | closed on | re-open only if |
|---|---|---|---|---|
| does a route loss through the builder (joint end-to-end fine-tune) beat continued route-head training? | no, three times (a - c between -0.07 and +0.007 in route success) | E202, E214, E232 | 2026-10-25 | a route loss that is not saturated on the training frames, or a builder that is not frozen at the start |
| do larger recall weights close the lanes-found gap at the 288 / 144 chain? | no beyond weight 3: weight 6 / range weighting give no Q gain (the weight buys lanes found, part of the Q gain is extra fine-tuning) | E219.rc1, rc2, rc4, f1eq | 2026-10-25 | a different mechanism for far range (input resolution, temporal accumulation) |
| do anchoring / fork-relocation settings add anything on top of fork pruning? | no | E215, E218 | 2026-10-25 | a new junction head |
| can three GPU jobs run at once on the RTX 3060? | no, they stall each other (100 iterations in 10 min) | loop 18 | 2026-10-25 | other hardware |
| does a matching-free Poisson set likelihood replacing the recall hinge lift Q or lanes found of the j1 fine-tune at an equal vertex budget? | no: Q -0.073 and lanes found -0.13 at 14.3 vertices per 100 m (one seed, full schedule, exact map); it is competitive only at 18.7 vertices | E244, E255 | 2026-10-27 | a calibration step for the validity scores before thinning, or the loss added next to the hinge |
