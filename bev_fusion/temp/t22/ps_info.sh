#!/bin/bash
# scratch: who are the other processes on the box
for p in "$@"; do
  echo "PID $p"
  readlink /proc/$p/cwd
  tr '\0' ' ' < /proc/$p/cmdline | cut -c1-250
  echo
done
