p=$1
ls -l /proc/$p/cwd
tr '\0' ' ' < /proc/$p/cmdline; echo
pp=$(ps -o ppid= -p $p | tr -d ' ')
for i in 1 2 3 4; do echo "parent $pp: $(tr '\0' ' ' < /proc/$pp/cmdline 2>/dev/null | cut -c1-220)"; pp=$(ps -o ppid= -p $pp | tr -d ' '); [ -z "$pp" ] && break; done
