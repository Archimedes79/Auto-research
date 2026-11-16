#!/bin/bash
# one-off: the window arithmetic of the launchers for some fake times
window_left_t() {  # d n
  d=$1; n=$2
  if [ $n -ge 1320 ]; then echo $((1440 - n + 420)); return; fi
  if [ $n -lt 420 ]; then echo $((420 - n)); return; fi
  if [ $d -le 5 ] && [ $n -ge 510 ] && [ $n -lt 1050 ]; then echo $((1050 - n)); return; fi
  echo 0
}
for t in "4 360" "4 440" "4 510" "4 900" "4 1049" "4 1100" "4 1320" "4 1380" "6 600" "6 1330" "7 100"; do set -- $t; echo "day $1 minute $2 -> $(window_left_t $1 $2) min left"; done
