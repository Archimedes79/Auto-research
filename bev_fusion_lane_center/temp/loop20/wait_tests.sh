#!/bin/bash
cd "$(dirname "$0")/../.."
for i in $(seq 1 360); do
  n=$(grep -c -E "^(PASS|FAIL)" temp/loop20/tests_out.txt)
  if [ "$n" -ge 14 ]; then break; fi
  sleep 10
done
cat temp/loop20/tests_out.txt
