p = "evaluation/jdup.py"
s = open(p, encoding="utf-8").read()
s = s.replace("partner_section=0, partner_link=0, conn_dup_with_c_partner=0)\n    dups", "partner_section=0, partner_link=0, conn_dup_with_c_partner=0, CBc=0)\n    dups", 1)
old = '''                t["conn_dup_with_c_partner"] += int(has_c)                                                 # a connector duplicate point that ALSO has a connector partner (hidden in CB when a B partner exists)
                if has_b:
'''
new = '''                t["conn_dup_with_c_partner"] += int(has_c)                                                 # ALL connector duplicate points that have a connector partner: the CC points (no B partner) and the CB points that also have one
                if has_b:
                    t["CBc"] += int(has_c)                                                                 # of the CB points: those that ALSO have a connector partner (hidden behind the B partner in the CB class)
'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace('"partner_section", "partner_link", "conn_dup_with_c_partner")\n    bins', '"partner_section", "partner_link", "conn_dup_with_c_partner", "CBc")\n    bins', 1)
old = '''CC (no B partner) {T['CC']} CB {T['CB']} (of which {T['conn_dup_with_c_partner']} connector duplicate points also have a connector partner) BC {T['BC']} BB {T['BB']}; "'''
new = '''CC (no B partner) {T['CC']} CB {T['CB']} BC {T['BC']} BB {T['BB']}; connector duplicate points with a connector partner {T['conn_dup_with_c_partner']} = CC + CB-with-a-connector-partner {T['CBc']} ({100 * T['conn_dup_with_c_partner'] / max(T['CC'] + T['CB'], 1):.1f} % of the connector duplicate points CC + CB); "'''
assert old in s; s = s.replace(old, new, 1)
old = '''    print("jdup selftest ok")'''
new = '''    # the connector-partner key: CC points count once (key == CC, CBc == 0), a lane with no connector near it has key 0, a connector pair beside a lane has both a B and a connector partner (CBc == key - CC)
    assert t3["conn_dup_with_c_partner"] == t3["CC"] and t3["CBc"] == 0, t3
    assert t["conn_dup_with_c_partner"] == 0 and t["CBc"] == 0, t
    pred4 = dict(pts=[lane(0.0, -30, 30), lane(30.0, -30, 30), lane(0.3, -20, 20), lane(-0.3, -20, 20)], edges=[], nodes=[[0, 1], [2, 3], [4, 5], [6, 7]])
    t4, _ = attribute(pred4, 2, ["section", "section"])
    assert t4["CB"] > 0 and t4["CBc"] == t4["CB"] and t4["conn_dup_with_c_partner"] == t4["CC"] + t4["CBc"] and t4["dup"] == TM.dup_counts(pred4)[0], t4
    print("jdup selftest ok")'''
assert old in s; s = s.replace(old, new, 1)
s = s.replace("CC (no B partner), CB (a B partner; `conn_dup_with_c_partner` of them also have a connector partner), BC, BB;", "CC (no B partner), CB (a B partner), BC, BB, and `conn_dup_with_c_partner` = the connector duplicate points that have a connector partner (CC + `CBc`, the CB points that also have one);", 1)
open(p, "w", encoding="utf-8").write(s); print("ok")
