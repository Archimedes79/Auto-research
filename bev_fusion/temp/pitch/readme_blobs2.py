"""inside commit_eval2.sh (GIT_INDEX_FILE is set): evaluation/README.md = HEAD's text + only MY lines of the working copy (it also holds other chats' uncommitted lines)"""
import subprocess

P = "bev_fusion/evaluation/README.md"
head = subprocess.run(["git", "show", "HEAD:" + P], capture_output=True, text=True, check=True).stdout.split("\n")
work = open(P, encoding="utf-8").read().split("\n")


def mine(prefixes):
    out = []
    for pre in prefixes:
        hit = [line for line in work if line.startswith(pre)]
        assert len(hit) == 1, (pre, len(hit))
        assert not any(line.startswith(pre) for line in head), ("already in HEAD", pre)
        out.append(hit[0])
    return out


def insert_before(lines, anchor_prefix, new):
    idx = [i for i, line in enumerate(lines) if line.startswith(anchor_prefix)]
    assert len(idx) == 1, (anchor_prefix, len(idx))
    return lines[:idx[0]] + new + lines[idx[0]:]


cmds = mine(["python evaluation/bev_suite.py runs/t1_r4_soa generalisation", "python evaluation/bev_suite.py runs/t1_r4_soa --curves eval", "python evaluation/bev_gtcheck.py [--n 24]"])
rows = mine(["| `bev_curves.py` |", "| `bev_gtcheck.py` |", "| `bev_response.py` |", "| `bev_mutants.py` |"])
head = insert_before(head, "python evaluation/bev_suite.py runs/t1_r4_soa quant run", cmds)
head = insert_before(head, "| `bev_legacy.py` |", rows)
text = "\n".join(head)
blob = subprocess.run(["git", "hash-object", "-w", "--stdin"], input=text, capture_output=True, text=True, check=True).stdout.strip()
subprocess.run(["git", "update-index", "--cacheinfo", f"100644,{blob},{P}"], check=True)
print("readme blob ok")
