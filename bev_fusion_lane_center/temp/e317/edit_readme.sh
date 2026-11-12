#!/bin/bash
# E317 scratch: append the E317 / E318 words to the README rows (sed, literal strings)
cd "$(dirname "$0")/../.."
sed -i '43s#Test: `python jends.py`. |$#E317 `--geom-w` (`geom_loss`): the delivered connector is supervised directly (the curve of every true keypoint pair, `_bezier_s` = the delivered Bezier with the learned handle scales, against the GT connector polyline; the keypoint ends against the GT ends), gradient into the heat, the headings and the scales; E318 `unc`: a position scale per keypoint (Laplace NLL, feature of the pair logit, output `kp_logsig`); both off by default (the E312 recipe is bit-identical). Test: `python jends.py`. |#' training/grid/README.md
sed -i '41s#Test: `python qjunction.py`. |$#`pair_polylines`: the GT connector polyline of every true pair (16 points) for the geometry loss of the pair head (E317). Test: `python qjunction.py`. |#' training/grid/README.md
sed -i '33s#`score --snap ...` as `qends_eval.py` |$#`score --snap ...` as `qends_eval.py`; E318: a run trained with `--unc` is loaded with its position scales and the dump carries `logsig` per keypoint |#' evaluation/README.md
