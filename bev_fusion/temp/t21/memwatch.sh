#!/usr/bin/env bash
# usage: memwatch.sh LIMIT_MB cmd...   runs cmd, kills it when its RSS passes LIMIT_MB, prints the peak RSS
limit=$1; shift
"$@" &
pid=$!
peak=0
while kill -0 $pid 2>/dev/null; do
  rss=$(awk '/VmRSS/ {print int($2/1024)}' /proc/$pid/status 2>/dev/null || echo 0)
  [ -n "$rss" ] && [ "$rss" -gt "$peak" ] && peak=$rss
  if [ "$peak" -gt "$limit" ]; then echo "KILLED: RSS $peak MB over $limit MB"; kill $pid; break; fi
  sleep 1
done
wait $pid; echo "exit $? peak RSS ${peak} MB"
