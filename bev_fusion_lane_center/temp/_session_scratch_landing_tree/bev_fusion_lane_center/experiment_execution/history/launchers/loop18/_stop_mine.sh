#!/bin/bash
# terminate processes of MY jobs whose command line contains the pattern (never other chats' jobs); usage: _stop_mine.sh <pattern>
for p in $(ps -eo pid,args | grep -F -- "$1" | grep -v grep | grep -v _stop_mine | awk '{print $1}'); do kill $p 2>/dev/null && echo "killed $p"; done
