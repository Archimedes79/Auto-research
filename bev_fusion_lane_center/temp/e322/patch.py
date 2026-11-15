import re
s = open("temp/e322/oracle_wiring.py", encoding="utf-8").read()
s = s.replace('rec = {v: [] for v in ("c", "o", "o_drop")}', 'rec = {v: [] for v in ("c", "o", "o_drop", "r2", "r3", "r6")}')
s = s.replace('stat = dict(ends=0, matched=0, found=0, changed=0)', 'stat = dict(ends=0, matched=0, found=0, both=0, same=0)')
s = s.replace('''        o_pairs.append(tuple(new))''', '''        o_pairs.append(tuple(new))''')
# agreement of the rule's choice at the keypoint with the oracle's choice at the GT end
s = s.replace('''                oe = oracle_end(G, gi, entry)
                if oe is not None:
                    stat["found"] += 1
                    new[slot], new[slot + 1] = oe[0], oe[1]''', '''                oe = oracle_end(G, gi, entry)
                if oe is not None:
                    stat["found"] += 1
                    ke = oracle_end(np.concatenate([qp[kidx + (0 if entry else k)], qd[kidx + (0 if entry else k)]])[None, :], 0, entry)      # the rule at the end the pair uses (keypoint, or the attached end)
                    if ke is not None:
                        stat["both"] += 1
                        stat["same"] += int(np.linalg.norm(ke[0] - oe[0]) < 1e-9)
                    new[slot], new[slot + 1] = oe[0], oe[1]''')
s = s.replace('''    for name, pp in (("c", pairs), ("o", o_pairs), ("o_drop", od_pairs)):
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pp, snap=True)''', '''    for name, pp, rr in (("c", pairs, None), ("o", o_pairs, None), ("o_drop", od_pairs, None), ("r2", pairs, 2.0), ("r3", pairs, 3.0), ("r6", pairs, 6.0)):
        raw, n_c, n_w = add_connectors(graph.lines, graph.edges, graph.nodes, bk["kind"], bk["piece_links"], np.asarray(pred["links"]["long"], bool), pp, snap=True, wire_pos=rr)''')
open("temp/e322/oracle_wiring.py", "w", encoding="utf-8").write(s)
