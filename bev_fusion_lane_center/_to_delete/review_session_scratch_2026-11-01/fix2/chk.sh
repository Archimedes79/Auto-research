for f in /tmp/ar_st_p1.txt /tmp/ar_st_p2.txt /tmp/ar_st_p3.txt /tmp/ar_st_e.txt /tmp/ar_selftest_d24c.txt; do
  echo "$f passed=$(grep -c 'autoresearch tools selftest passed' $f)"
  grep -v '^note: S.c' $f | grep -m4 -i 'error\|assert\|Traceback'
done
