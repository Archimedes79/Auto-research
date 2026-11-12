#!/bin/bash
# The recipe the simplification arms E331 - E335 remove ONE thing from (sourced by the start scripts and the chains of temp/e331 ... temp/e334).
# RE-BASING (D85): when E320 (the differentiable attachment) is KEEP and becomes the reference, change ONLY this file and chain_lib.sh:
#   RECIPE_FLAGS   + the training flags of the E320 arm (jends_train.py: e.g. --geom-w 0.5 and E320's attach flag; read temp/e320/start_pairs.sh)
#   CONTROL_PREFIX = e320_a_s (the E320 arm of the same seed is the control; jrule_removal.py --control $CONTROL_PREFIX)
#   BASE_MAPS_PREFIX stays e307_n_s unless E320 trains on other maps; the maps stages of E332 / E333 (own maps) keep MAPS_FLAGS
#   RULE_READOUTS="attach" and READOUT_EXTRA="--attach" (post_pair then scores E320's attached read-out next to the unsnapped and the snapped one; the rule judges on the attached one)
# and re-run the cards' controls from E320's runs, nothing else in the start scripts.
export BFLC_SIG=1.0
MAPS_FLAGS="--ctx dil --sigma 1.0 --steps 3000 --workers 3"              # the E307 maps stage (jmaps_train.py)
RECIPE_FLAGS="--e2e --scales --maps-w 1.0 --steps 3000 --workers 3"      # the E312 pair stage (jends_train.py)
BASE_MAPS_PREFIX=e307_n_s                                                # the maps of the E312 recipe (the seed is appended): runs/e307_n_s<S>
CONTROL_PREFIX=e312_p_s                                                  # the control run of the same seed
RULE_READOUTS="plain snap"                                             # the read-outs jrule_removal.py judges on: plain = unsnapped (differentiable), snap = the rule step; re-based on E320: "attach"
READOUT_EXTRA=""                                                         # re-based on E320: "--attach" (post_pair then also scores the attached read-out)
