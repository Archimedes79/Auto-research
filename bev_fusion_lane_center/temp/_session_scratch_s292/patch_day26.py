import io
p = "temp/day26/day26.sh"
s = io.open(p, encoding="utf-8", newline="").read()
a = '''# quiet: wait (up to 12 h) until no CPU scoring job of the E345 chain runs and 8 GB of RAM are free
quiet() {
  local t=0
  while { pgrep -f "jprune_table.py score" > /dev/null || [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 8000 ]; }; do'''
b = '''# quiet: wait (up to 12 h) until 8 GB of RAM are free (the lane lead put E343 first, 2026-11-15 03:20: the E345 scoring jobs are not waited for; one heavy job of this chain at a time)
quiet() {
  local t=0
  while [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt 8000 ]; do'''
assert s.count(a) == 1
s = s.replace(a, b)
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("ok")
