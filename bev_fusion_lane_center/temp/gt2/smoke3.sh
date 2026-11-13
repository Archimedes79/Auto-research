#!/bin/bash
cd "$(dirname "$0")/../.."
bash temp/e307/start_narrow.sh 0 smoke > temp/gt2/smoke3.log 2>&1; echo "E307 smoke exit $? $(date)" >> temp/gt2/smoke3.log
