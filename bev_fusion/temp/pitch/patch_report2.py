p = "../autoresearch/tools/run_report.py"
t = open(p, encoding="utf-8").read()
def rep(a, b):
    global t
    assert t.count(a) == 1, (a[:70], t.count(a))
    t = t.replace(a, b)

rep("def read_table(path, limit=40):", "def read_table(path, limit=None):")
rep('''    """eval/table.txt: 'key   value' per line -> [(key, float)] (first `limit` lines)"""''', '''    """eval/table.txt: 'key   value' per line -> [(key, float)] (all lines, or the first `limit`)"""''')
rep("            if len(out) >= limit:", "            if limit and len(out) >= limit:")
# gap block with explicit table paths + the caveat of the pretrained heads
rep("def gap_block(rundir, card):", "def gap_block(rundir, card, val_path=None, train_path=None):")
rep('''    tr, va = read_table(os.path.join(rundir, "eval", "table_train.txt")), dict(read_table(os.path.join(rundir, "eval", "table.txt")))''',
    '''    tr = read_table(train_path or os.path.join(rundir, "eval", "table_train.txt"))
    va = dict(read_table(val_path or os.path.join(rundir, "eval", "table.txt")))''')
rep('''"spread over the scenes of each split. The TEST split is not used here (it needs a logged row under a recorded decision).</p>")''',
    '''"spread over the scenes of each split. The TEST split is not used here (it needs a logged row under a recorded decision). Caveat: the frozen image heads "
            "(pre-trained DualBEV / BEVFusion) saw the nuScenes train scenes too, so part of a train-val gap can come from them and not from the trainable network.</p>")''')
rep("    h.append(gap_block(rundir, card))", "    h.append(gap_block(rundir, card, GAP.get('val'), GAP.get('train')))")
rep('OPS = {"fps": 10.0,', 'GAP = {}                                           # explicit table paths of the train-vs-val block (bev_suite generalisation), else eval/table.txt and eval/table_train.txt\nOPS = {"fps": 10.0,')
# numbers: scrollable, NaN-safe delta
rep('''        h.append("<table><tr><th class='l'>metric</th>" + "".join(''', '''        h.append("<div style='max-height:640px;overflow:auto;display:inline-block'><table><tr><th class='l'>metric</th>" + "".join(''')
rep('''            if ref and key in ref:
                diff = v - ref[key]''', '''            if ref and key in ref and v == v and ref[key] == ref[key]:
                diff = v - ref[key]''')
rep('''        h.append("</table><p class='note'>green = better than the first reference''', '''        h.append("</table></div><p class='note'>green = better than the first reference''')
# review key per page path
rep("const KEY='review:'+document.title;", "const KEY='review:'+location.pathname+':'+document.title;")
# ingest hardening
rep('''    d = json.load(open(src, encoding="utf-8"))
    out = os.path.join(rundir, "report", "review.json")''', '''    d = json.load(open(src, encoding="utf-8"))
    name = os.path.basename(os.path.normpath(rundir))
    if d.get("run") not in (None, name):
        raise SystemExit(f"this review is of run '{d.get('run')}', not of '{name}': not filed")
    out = os.path.join(rundir, "report", "review.json")''')
rep('os.replace(out, out + time.strftime(".%Y%m%d_%H%M%S.old"))', 'os.replace(out, out + time.strftime(".%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.old")')
rep("overall: {d.get('overall', '')[:300]}", "overall: {(d.get('overall') or '')[:300]}")
rep('''    ap.add_argument("--ingest", default="", help="a review.json exported from the page")''', '''    ap.add_argument("--ingest", default="", help="a review.json exported from the page")
    ap.add_argument("--val-table", default="", help="table of the validation numbers for the train-vs-val block (default eval/table.txt)")
    ap.add_argument("--train-table", default="", help="table of the train numbers (default eval/table_train.txt)")''')
rep('''    OPS.update(fps=a.fps,''', '''    GAP.update(val=a.val_table or None, train=a.train_table or None)
    OPS.update(fps=a.fps,''')
# latency: show whether it was idle
rep('''"<p class='note'>ms per frame as measured by the evaluation; valid only if it ran on an idle GPU (eval/latency.json records the load).</p>")''',
    '''f"<p class='note'>ms per frame as measured by the evaluation; valid only if it ran on an idle GPU (recorded: idle = {lat.get('idle', 'not recorded')}).</p>")''')
open(p, "w", encoding="utf-8", newline="\n").write(t)
print("run_report patched")
