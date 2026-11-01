#!/usr/bin/env bash
# the link_views.sh guard before the fix: a file opened through the mini view is held open by this shell (fd 3)
set -euo pipefail
exec 3< "$HOME/data/nuscenes/LICENSE"
if find /proc/[0-9]*/fd -lname "$HOME/data/*" 2>/dev/null | grep -q .; then echo "old guard: FIRES"; else echo "old guard: does not fire"; fi
echo "fd 3 resolves to: $(readlink /proc/$$/fd/3)"
exec 3<&-
