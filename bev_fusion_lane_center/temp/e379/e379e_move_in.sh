#!/bin/bash
# move the dev copies of jtoken_eval.py and its test into evaluation/ (atomic rename), only when the live files are still the ones the dev copies were made from (temp/e379/e379e_base.sha256)
set -e
cd "$(dirname "$0")/../.."
D=temp/e379/e379e_session_scratch_dev/evaluation
want_py=$(grep 'jtoken_eval.py$' temp/e379/e379e_base.sha256 | grep -v tests | cut -c1-64)
want_t=$(grep 'tests/test_jtoken_eval.py$' temp/e379/e379e_base.sha256 | cut -c1-64)
got_py=$(sha256sum evaluation/jtoken_eval.py | cut -c1-64)
got_t=$(sha256sum evaluation/tests/test_jtoken_eval.py | cut -c1-64)
[ "$want_py" = "$got_py" ] || { echo "STOP: evaluation/jtoken_eval.py changed since the copy ($got_py, copy of $want_py)"; exit 1; }
[ "$want_t" = "$got_t" ] || { echo "STOP: evaluation/tests/test_jtoken_eval.py changed since the copy ($got_t, copy of $want_t)"; exit 1; }
new_py=$(sha256sum $D/jtoken_eval.py | cut -c1-64)
new_t=$(sha256sum $D/tests/test_jtoken_eval.py | cut -c1-64)
mv $D/jtoken_eval.py evaluation/jtoken_eval.py
mv $D/tests/test_jtoken_eval.py evaluation/tests/test_jtoken_eval.py
[ "$(sha256sum evaluation/jtoken_eval.py | cut -c1-64)" = "$new_py" ] && [ "$(sha256sum evaluation/tests/test_jtoken_eval.py | cut -c1-64)" = "$new_t" ]
rmdir $D/tests $D temp/e379/e379e_session_scratch_dev
echo "moved in: jtoken_eval.py $want_py -> $new_py, test_jtoken_eval.py $want_t -> $new_t"
