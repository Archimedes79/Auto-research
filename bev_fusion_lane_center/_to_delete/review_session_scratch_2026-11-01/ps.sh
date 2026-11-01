for p in $(pgrep -f "python|bash temp|bash tests"); do echo "== $p cwd=$(readlink /proc/$p/cwd) ppid=$(ps -o ppid= -p $p)"; tr "\0" " " < /proc/$p/cmdline; echo; done
