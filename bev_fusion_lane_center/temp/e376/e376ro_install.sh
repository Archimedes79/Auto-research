#!/bin/bash
# move the E376 read-out additions into place: a copy next to the target, then an atomic rename (other chats' read-outs import jtoken_eval.py); refuses when a target changed since the scratch copies were made
set -e
cd "$(dirname "$0")/../.."
cmp -s evaluation/jtoken_eval.py temp/e376/e376ro_jtoken_eval.orig.py || { echo "evaluation/jtoken_eval.py changed since the copy: stop"; exit 1; }
cmp -s evaluation/tests/test_jtoken_eval.py temp/e376/e376ro_test_jtoken_eval.orig.py || { echo "evaluation/tests/test_jtoken_eval.py changed since the copy: stop"; exit 1; }
cp temp/e376/e376ro_jtoken_eval.py evaluation/.jtoken_eval.py.e376tmp
mv -f evaluation/.jtoken_eval.py.e376tmp evaluation/jtoken_eval.py
cp temp/e376/e376ro_test_jtoken_eval.py evaluation/tests/.test_jtoken_eval.py.e376tmp
mv -f evaluation/tests/.test_jtoken_eval.py.e376tmp evaluation/tests/test_jtoken_eval.py
cmp evaluation/jtoken_eval.py temp/e376/e376ro_jtoken_eval.py && cmp evaluation/tests/test_jtoken_eval.py temp/e376/e376ro_test_jtoken_eval.py && echo installed
