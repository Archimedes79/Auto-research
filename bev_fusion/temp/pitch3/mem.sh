grep -E "MemAvailable|MemTotal|SwapFree" /proc/meminfo
pgrep -af "python" | cut -c1-110 | head -8
