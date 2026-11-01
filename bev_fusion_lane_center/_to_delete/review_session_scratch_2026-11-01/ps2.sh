p=93978; while [ "$p" -gt 1 ]; do echo "$p: $(tr '\0' ' ' < /proc/$p/cmdline)"; p=$(ps -o ppid= -p $p | tr -d ' '); done
tr '\0' '\n' < /proc/93978/environ | grep -iE "cfg|bf_|pythonpath|run" 
