#!/bin/bash
# D113 / E365: a second build of the first 40 VAL frames of B' seed 0 (K = 512) must equal the first 40 frames of the full cache in EVERY field (tok, f2, uout, uin, logits too): the determinism fixture
cd "$(dirname "$0")/../.."
bash tools/env/_pyr.sh lane_tokcache.py build --run runs/e292_a_s0 --split val --K 512 --limit 40 --workers 2 || exit 1
bash tools/env/_py14.sh lane_tokcache.py same --a runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1/val --b runs/e365_cache/be292_a_0_K512_9ad1d4ce30_v1_first40/val --n 40
