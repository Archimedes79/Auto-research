cd "$1"
for i in 1 2 3; do (nice python3 ../autoresearch/tools/selftest.py > /tmp/ar_st_p$i.txt 2>&1; echo "EXIT $?" >> /tmp/ar_st_p$i.txt) & done
wait
for i in 1 2 3; do tail -1 /tmp/ar_st_p$i.txt; grep -v '^note: S.c' /tmp/ar_st_p$i.txt | grep -B14 -A2 'Traceback' | head -34; done
