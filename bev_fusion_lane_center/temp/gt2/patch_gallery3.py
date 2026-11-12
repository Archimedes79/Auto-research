p = "temp/jr/fail_gallery.py"
s = open(p, encoding="utf-8").read()


def sub(old, new):
    global s
    assert old in s, old[:70]
    s = s.replace(old, new, 1)


sub('OUT = os.path.join(ROOT, "temp", "jr", "media", "errors")', 'SNAP = os.environ.get("GAL_SNAP", "0") == "1"                                     # E311: the snapped read-out\nOUT = os.path.join(ROOT, "temp", "jr", "media", os.environ.get("GAL_OUT", "errors"))')
sub("QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), prs)", "QEV.add_connectors(graph.lines, graph.edges, graph.nodes, bk[\"kind\"], bk[\"piece_links\"], np.asarray(pred[\"links\"][\"long\"], bool), prs, snap=SNAP)")
open(p, "w", encoding="utf-8").write(s)
print("ok")
