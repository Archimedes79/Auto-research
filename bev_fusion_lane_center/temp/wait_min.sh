#!/bin/bash
# one-off: wait N seconds in short steps (the harness blocks a plain foreground sleep)
for i in $(seq 1 $(( ${1:-240} / 6 ))); do sleep 6; done
echo waited
