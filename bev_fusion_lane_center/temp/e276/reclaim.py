# scratch: the WSL VM has 188 MB free and 22 GB page cache: the DataLoader workers of both E279 jobs died with "page allocation failure: order:4" (a 64 KB 9p read buffer) at 04:18 and 04:32.
# Touching 14 GB in 1 GB steps makes the kernel evict page cache; freeing it again gives large free blocks back. User level, no root; the trainings keep running (their RSS is a few GB).
import time
chunks = []
for i in range(14):
    chunks.append(bytearray(1 << 30))
    time.sleep(0.5)
    print("held", i + 1, "GB", flush=True)
time.sleep(2)
chunks.clear()
print("released", flush=True)
