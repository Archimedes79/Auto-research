import subprocess
t = subprocess.run(["git", "-C", "/mnt/c/Auto-research", "show", "HEAD:autoresearch/README.md"], capture_output=True, text=True).stdout
for k, l in enumerate(t.split("\n")):
    if l.startswith("| `") and ("suite" in l[:20] or "trainkit" in l[:20] or "history" in l[:20] or "suitecli" in l[:20] or "runner" in l[:20]):
        print(k, repr(l[:90]))
