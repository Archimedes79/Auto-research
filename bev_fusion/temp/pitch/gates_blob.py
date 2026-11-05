"""inside commit_eval2.sh (GIT_INDEX_FILE is set): tests/run_gates.sh = HEAD's text + only MY block (curve metrics, GT against GT, response test); the working copy lacks
rows that other chats committed (radar encoder, radar slot). Also writes the merged text over the working copy so that it stops lagging behind HEAD."""
import subprocess

P = "bev_fusion/tests/run_gates.sh"
head = subprocess.run(["git", "show", "HEAD:" + P], capture_output=True, text=True, check=True).stdout.split("\n")
work = open(P, encoding="utf-8").read().split("\n")
i0 = [i for i, line in enumerate(work) if line.startswith('run "curve metrics"')]
assert len(i0) == 1
i1 = i0[0]
while not work[i1].startswith("fi"):
    i1 += 1
block = work[i0[0]:i1 + 1]
assert any("bev_gtcheck" in line for line in block) and any("bev_response" in line for line in block), block
assert not any("bev_curves" in line for line in head), "already in HEAD"
idx = [i for i, line in enumerate(head) if line.startswith('run "orientation"')]
assert len(idx) == 1, idx
merged = head[:idx[0]] + block + head[idx[0]:]
text = "\n".join(merged)
blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=text, capture_output=True, text=True, check=True).stdout.strip()
subprocess.run(["git", "update-index", "--add", "--cacheinfo", f"100755,{blob},{P}"], check=True)
open(P, "w", encoding="utf-8", newline="\n").write(text)
print("run_gates blob ok:", len(block), "lines of mine,", len(head), "lines of HEAD kept")
