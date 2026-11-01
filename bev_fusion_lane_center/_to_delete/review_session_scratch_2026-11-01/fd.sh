exec 7< ~/data/nuscenes/LICENSE
readlink /proc/$$/fd/7
find /proc/$$/fd -lname "$HOME/data/*" 2>/dev/null | wc -l
