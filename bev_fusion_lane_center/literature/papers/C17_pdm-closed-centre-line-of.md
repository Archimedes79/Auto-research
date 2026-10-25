# C17 - PDM-Closed: centre-line of the route lane sequence + simulation scoring (older anchor, 2023)

*Pass: loop 17 (2026-10-24), file `C_map_matching_planner.md`, cluster `map_matching_planner`. Depth M here means: method sections queried through a page digest (see sources/lit17/C_map_matching_planner.md, section 1).*

- papers: Parting with Misconceptions about Learning-based Vehicle Motion Planning (Dauner et al., CoRL 2023) https://arxiv.org/abs/2306.07962
- depth: M
- stage: S10, E
- idea: Graph search finds the lane sequence along the route, its centre-line feeds IDM for the longitudinal profile; proposals at 5 speeds x 3 lateral offsets (-1, 0, +1 m), 4 s at 10 Hz, LQR + bicycle simulation, best proposal extended to 8 s, emergency brake. PDM-Hybrid adds a learned offset MLP beyond 2 s.
- results (Val14, CLS-R / CLS-NR / OLS / ms): PDM-Closed 92 / 93 / 44 / 91 ms; PDM-Hybrid 92 / 93 / 84 / 96 ms; without lateral offsets CLS-R 89 at 55 ms. Route / centre-line errors not analysed; no lane changes (collision risk) and needs HD maps.
- differentiable: no (rule + simulation). 
- fit_to_us: baseline - effort S - the strong "route centre-line is nearly the plan" prior: a baseline planner for our route-labelled graph would be: take the lane chain with highest route probability, offset it by learned residual; our 0.34 m path error should be compared with it.
- tried_before: oracle route planner reference 0.316 m (project note). verified: partly
