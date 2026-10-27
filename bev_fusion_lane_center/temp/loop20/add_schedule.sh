#!/bin/bash
cd "$(dirname "$0")/../.."
S="python3 ../autoresearch/tools/schedule.py add"
$S E252.a0 --node I1 --kind analysis --resource cpu --budget 10m --note "map offset by city (done)" --runs E252.a0
$S E252.b0 --node I1 --kind analysis --resource cpu --budget 10m --depends E252.a0 --note "referee follow-up" --runs E252.b0
$S E240.a3 --node I2 --kind train --resource gpu --budget 20m --note "seed pair 3, j1rc, stored map" --runs E240.a3
$S E240.c3 --node I2 --kind train --resource gpu --budget 20m --depends E240.a3 --note "seed pair 3, j1eq, stored map" --runs E240.c3
$S E255.mc --node I3 --kind analysis --resource cpu --budget 15m --note "matched-vertex grids of the exact-map controls" --runs E255.mc
$S E255.a0 --node I3 --kind train --resource gpu --budget 25m --depends E240.c3 --note "Poisson set loss, full schedule" --runs E255.a0
$S E255.mp --node I3 --kind analysis --resource cpu --budget 15m --depends E255.a0 --note "matched-vertex procedure for arm a (python e255_matched.py e25_j1phd2_s0)"
$S E253.t --node I4 --kind timing --resource gpu --budget 60m --depends E255.mp --note "exclusive timing window; tell the BEV-system chat when done"
python3 ../autoresearch/tools/schedule.py show | cut -c1-160
