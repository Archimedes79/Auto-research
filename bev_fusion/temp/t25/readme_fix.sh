set -e
f=tests/README.md
sed -i '55s|bit for bit, in a key frame and the next; awake it differs\.|bit for bit, in a key frame and the next, the carried radar buffer is the radar block'"'"'s output on the newest sweep, and with `delta_slots` + the motion block `radar_side` is the radar block'"'"'s on `radar_seq[0]` and the no-memory arm'"'"'s; awake it differs.|' $f
sed -i '55s|D: the write mask is made per frame from that frame'"'"'s own sweep\.|D: the write mask is made per frame from that frame'"'"'s own sweep, and the radar block gets the camera block'"'"'s ego step (v, omega) and past chains at every frame.|' $f
sed -i '55s|G: 5 planted bugs\.|G: 9 planted bugs.|' $f
sed -i '55s|about 100 s under load): 5 / 5 killed (radar carry ignored, radar frames in the wrong order, masks in the wrong order, radar state not carried, fused state cut by a detach)|RESULT_PLACEHOLDER|' $f
sed -i '91s|5 mutants|9 mutants|' tests/run_gates.sh
