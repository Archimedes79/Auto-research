# E56: Direction consistency as an additional acceptance criterion of the connectors: does the predicted travel direction reveal the false ones?

Status: completed

**Question:** The false connectors of E53 raise the dense heat along their curve almost as much as the true ones. The dense head also predicts the direction of travel (dx, dy) in every cell, which the heat threshold ignores. Does requiring that the tangent of a connector agrees with the predicted direction along the curve reject false connectors without losing true ones - a better operating point than the heat threshold alone? (Pixel-based direction field, vector-level connector, rule on top; no new training.)

## Setup
- Rule on top of E53: for every candidate connector (cubic Bezier curve from a split key point to a merge key point that passes the heat threshold tau_h) the cosine between the curve tangent (in travel direction) and the predicted direction field is sampled bilinearly at the interior points of the curve (cells with a predicted vector shorter than 0.1 are ignored) and averaged along the curve; the connector is accepted only if this mean cosine is >= cos_c. No learned parameter. Convention check (check_dircos.py): along the ground-truth centerlines the predicted direction has a mean cosine of 0.908 with the true direction (95 % of the points above 0.5), so the sign and the axes are right.
- Sweep on VAL: tau_h 0.1 and 0.5, cos_c 0.3 / 0.6 / 0.8 (the left-most point of the figure is E53 without the criterion). The middle setting (tau_h 0.5, cos_c 0.6) was fixed beforehand and then run on VAL and TEST; paired scene-bootstrap intervals of its differences are given in E54.

![architecture](E56_architecture.png)

## Numbers

**Connector acceptance with the predicted travel direction as an additional criterion**

| variant | set | F1@1m | P / R @1m | route rec / prec / F | split F1 3 / 6 m | split P / R @6m | merge F1 3 / 6 m | merge P / R @6m | dir acc | VAL selection score | path F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| previous final system (no key points) | VAL | 0.899 | 0.924 / 0.875 | 0.773 / 0.827 / 0.799 | 0.216 / 0.308 | 0.375 / 0.262 | 0.253 / 0.417 | 0.501 / 0.358 | 0.964 | 4.123 | 0.560 |
| previous final system (no key points) | TEST | 0.876 | 0.910 / 0.844 | 0.710 / 0.783 / 0.744 | 0.219 / 0.342 | 0.392 / 0.304 | 0.215 / 0.380 | 0.455 / 0.327 | 0.953 | - | 0.484 |
| DEFAULT key-point nodes (E44) | VAL | 0.899 | 0.925 / 0.876 | 0.755 / 0.836 / 0.793 | 0.454 / 0.497 | 0.739 / 0.375 | 0.545 / 0.603 | 0.821 / 0.477 | 0.964 | 4.486 | 0.533 |
| DEFAULT key-point nodes (E44) | TEST | 0.876 | 0.910 / 0.844 | 0.683 / 0.791 / 0.733 | 0.502 / 0.552 | 0.825 / 0.415 | 0.537 / 0.607 | 0.875 / 0.464 | 0.953 | - | 0.459 |
| connectors, heat >= 0.10, no direction criterion (E53) | VAL | 0.898 | 0.913 / 0.884 | 0.753 / 0.805 / 0.778 | 0.655 / 0.702 | 0.783 / 0.635 | 0.593 / 0.646 | 0.672 / 0.622 | 0.959 | 4.700 | - |
| connectors, heat >= 0.50, no direction criterion (E53) | VAL | 0.899 | 0.916 / 0.883 | 0.754 / 0.810 / 0.781 | 0.644 / 0.689 | 0.781 / 0.615 | 0.595 / 0.648 | 0.691 / 0.611 | 0.959 | 4.698 | 0.536 |
| connectors, heat >= 0.10, direction cosine >= 0.3 | VAL | 0.898 | 0.914 / 0.883 | 0.754 / 0.809 / 0.780 | 0.654 / 0.699 | 0.787 / 0.628 | 0.594 / 0.646 | 0.684 / 0.612 | 0.961 | 4.702 | - |
| connectors, heat >= 0.10, direction cosine >= 0.6 | VAL | 0.899 | 0.916 / 0.882 | 0.754 / 0.816 / 0.784 | 0.636 / 0.680 | 0.787 / 0.600 | 0.584 / 0.640 | 0.703 / 0.587 | 0.963 | 4.686 | - |
| connectors, heat >= 0.10, direction cosine >= 0.8 | VAL | 0.899 | 0.917 / 0.881 | 0.754 / 0.819 / 0.785 | 0.629 / 0.674 | 0.787 / 0.589 | 0.578 / 0.636 | 0.709 / 0.577 | 0.964 | 4.679 | - |
| connectors, heat >= 0.50, direction cosine >= 0.3 | VAL | 0.900 | 0.917 / 0.882 | 0.754 / 0.812 / 0.782 | 0.642 / 0.685 | 0.786 / 0.607 | 0.596 / 0.649 | 0.705 / 0.600 | 0.961 | 4.697 | - |
| connectors, heat >= 0.50, direction cosine >= 0.6 | VAL | 0.900 | 0.919 / 0.881 | 0.755 / 0.817 / 0.785 | 0.625 / 0.668 | 0.786 / 0.581 | 0.586 / 0.643 | 0.723 / 0.579 | 0.963 | 4.680 | - |
| connectors, heat >= 0.50, direction cosine >= 0.8 | VAL | 0.900 | 0.920 / 0.881 | 0.755 / 0.819 / 0.786 | 0.617 / 0.660 | 0.785 / 0.570 | 0.580 / 0.639 | 0.729 / 0.569 | 0.964 | 4.669 | - |
| PRE-SET middle setting: heat >= 0.5, direction cosine >= 0.6 (VAL + TEST) | VAL | 0.900 | 0.919 / 0.881 | 0.755 / 0.817 / 0.785 | 0.625 / 0.668 | 0.786 / 0.581 | 0.586 / 0.643 | 0.723 / 0.579 | 0.963 | 4.680 | 0.539 |
| PRE-SET middle setting: heat >= 0.5, direction cosine >= 0.6 (VAL + TEST) | TEST | 0.876 | 0.905 / 0.848 | 0.680 / 0.773 / 0.724 | 0.643 / 0.688 | 0.847 / 0.580 | 0.552 / 0.621 | 0.741 / 0.534 | 0.952 | - | 0.466 |

## Result images

![Direction-consistency criterion for the connectors (VAL): split F1, merge F1 and route F against the minimum cosine between the curve tangent and the predicted direction field; left-most point: criterion off (E53).](E56_e56_sweep.png)
*Direction-consistency criterion for the connectors (VAL): split F1, merge F1 and route F against the minimum cosine between the curve tangent and the predicted direction field; left-most point: criterion off (E53).*

## Findings
- The criterion removes some false merges but mostly recall. VAL at tau_h 0.5: merge precision 0.691 -> 0.705 / 0.723 / 0.729 for cos_c 0.3 / 0.6 / 0.8, split precision unchanged (0.781 -> 0.786 / 0.786 / 0.785), recall falls (split 0.615 -> 0.607 / 0.581 / 0.570, merge 0.611 -> 0.600 / 0.579 / 0.569), route F rises 0.781 -> 0.782 / 0.785 / 0.786 and the point F1 stays at 0.899-0.900; the same pattern at tau_h 0.1 (route F 0.778 -> 0.780 / 0.784 / 0.785, split F1 0.702 -> 0.699 / 0.680 / 0.674). The VAL selection score (split + merge F1 + 2 route F + 2 point F1) is 4.70 for cos_c 0.3 and falls for stricter values (4.68 / 4.67), i.e. the criterion is not selected.
- TEST, middle setting (tau_h 0.5, cos_c 0.6): split / merge F1 (6 m) 0.688 / 0.621 (precision 0.847 / 0.741, recall 0.580 / 0.534), route F 0.724, point F1 0.876, path F1 0.466. Against the connectors without the criterion (E53, tau_h 0.5) the paired differences are route F +0.005 [0.002, 0.008], split F1 -0.026 [-0.043, -0.012] and merge F1 -0.016 [-0.029, -0.003]; against the DEFAULT of E44 route F -0.009 [-0.014, -0.005], split F1 +0.136 [0.089, 0.179], merge F1 +0.014 [-0.011, +0.038] (E54).
- Read as an operating-point trade-off, the criterion regains connectivity at the cost of node recall in a slightly cheaper exchange than the heat threshold (TEST: about 0.005 split F1 given up per 0.001 route F regained, against about 0.008 when the heat threshold is raised from 0.5 to 0.6) - a difference of the size of the intervals, i.e. not a clear improvement. The predicted travel direction therefore carries little information for the connector decision beyond the heat of the same head; the false connectors do not seem to run against the predicted direction but to join the wrong pair of lanes (a reading that was not tested separately). The heat threshold of E53 stays the switch between the operating points; the direction criterion is documented as an optional second switch.

## Limits
- Rule-based and tied to the dense head: the direction field has the same errors as the heat (same network, same unseen-scene shift); three cos_c and two tau_h values on a 22-scene VAL, the middle setting was not tuned; single seed. A learned decision on connector features (heat statistics, direction agreement, key-point scores, geometry) is the principled next step and was not tried.
