import subprocess
for path, nums in (("bev_fusion/training/README.md", (22, 26, 33)), ("bev_fusion/tests/README.md", (13, 14))):
    head = subprocess.run(["git", "show", "HEAD:" + path], capture_output=True, text=True, cwd="/mnt/c/Auto-research", encoding="utf-8").stdout.splitlines()
    for n in nums:
        print(path.split("/")[-2], n, "HEAD tail:", head[n - 1][-260:])
        print()
