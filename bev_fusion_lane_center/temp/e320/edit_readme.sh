#!/bin/bash
# E320 scratch: append the E320 words to the README rows (sed, literal strings)
cd "$(dirname "$0")/../.."
sed -i '43s#both off by default (the E312 recipe is bit-identical). Test: `python jends.py`. |$#E320 `--attach` (`Attach`): the connector ends are attached to the nodes of B'"'"'s frozen node head (softmax over the nodes in reach + a dustbin, the end = the weighted mean, so the curve touches a delivered piece end by construction; a differentiable replacement of the snap), trained by the geometry loss on the FINAL curve + a cross entropy over the candidates, dump keys `att_pos / att_dir / att_p`; all off by default (the E312 recipe is bit-identical). Test: `python jends.py`. |#' training/grid/README.md
sed -i '33s#the dump carries `logsig` per keypoint |$#the dump carries `logsig` per keypoint; E320: `--attach` loads an attach run, the dump carries `att_pos / att_dir / att_p`, `score --attach` draws the connectors from the attached ends without the snap (`ends_card_attach_thr<t>.json`) |#' evaluation/README.md
sed -i '31s#|$#|#' evaluation/README.md
