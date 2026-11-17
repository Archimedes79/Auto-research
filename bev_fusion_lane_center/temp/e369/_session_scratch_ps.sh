for p in 36133 36179; do
  tr '\0' '\n' < /proc/$p/cmdline | grep -A1 -E '^--(after|id)$' | tr '\n' ' '
  echo
  tr '\0' ' ' < /proc/$p/cmdline | grep -o -- ' -- bash.*' | cut -c1-250
done
