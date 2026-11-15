for p in $(pgrep -f "[j]ends_train.py --name e312_p_s0"); do
  if tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "BFLC_SIG=1.0"; then echo "pid $p: BFLC_SIG=1.0 present"; else echo "pid $p: no BFLC_SIG"; fi
done
