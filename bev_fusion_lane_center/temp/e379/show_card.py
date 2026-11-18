"""print the paired-vs-ref block of a token card and the head's own numbers (scratch, E379 read-out)  usage: python show_card.py <card.json> [<block>]"""
import json, sys
d = json.load(open(sys.argv[1]))
blk = sys.argv[2] if len(sys.argv) > 2 else "paired_vs_ref_run"
m = d["metrics"]
print("cell", d["info"].get("decode_variant"))
print("own  topo_F1 %.4f  junction_F1 %.4f  F1@1.0 %.4f  lanes_found %.4f  dup_frac %.4f  Q %.4f  topo_P %.4f  topo_R %.4f  junction_P %.3f junction_R %.3f" % tuple(m[k] for k in ("topo_F1", "junction_F1", "F1@1.0", "lanes_found", "dup_frac", "Q", "topo_P", "topo_R", "junction_P", "junction_R")))
for k, v in d["info"][blk].items():
    if isinstance(v, dict) and "diff" in v:
        print("%-14s a %.4f  b %.4f  diff %+.4f [%+.4f, %+.4f]  %s" % (k, v["a"], v["b"], v["diff"], v["lo"], v["hi"], v.get("verdict", "")))
