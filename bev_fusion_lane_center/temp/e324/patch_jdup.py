p = "evaluation/jdup.py"
s = open(p, encoding="utf-8").read()
# 1. attribute(): count, for a connector duplicate point with a B partner, whether a connector partner exists as well (CBc), and CC only when NO B partner is in range
old = '''        for j in nb:
            if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > cos_min:
                t["dup"] += 1
                t[("C" if isC[k] else "B") + ("C" if isC[j] else "B")] += 1
                if isC[k]:
                    dups[int(I[k]) - nB] += 1
                    if not isC[j]:
                        for lo, hi, name in ARC:
                            if lo <= margin[k] < hi:
                                t[name] += 1
                        if kinds is not None:
                            t["partner_link" if kinds[int(I[j])] == "link" else "partner_section"] += 1
                break
    return t, (dups, pts)'''
new = '''        partners = [j for j in nb if I[j] != I[k] and (int(I[k]), int(I[j])) not in near and abs(float(D[k] @ D[j])) > cos_min]
        if partners:
            has_c = any(isC[j] for j in partners)
            has_b = any(not isC[j] for j in partners)
            j = next((j for j in partners if not isC[j]), partners[0])                                      # the B partner when there is one (the indices of B are the lower ones, as in dup_counts' first hit)
            t["dup"] += 1
            t[("C" if isC[k] else "B") + ("C" if isC[j] else "B")] += 1
            if isC[k]:
                dups[int(I[k]) - nB] += 1
                t["conn_dup_with_c_partner"] += int(has_c)                                                 # a connector duplicate point that ALSO has a connector partner (hidden in CB when a B partner exists)
                if has_b:
                    for lo, hi, name in ARC:
                        if lo <= margin[k] < hi:
                            t[name] += 1
                    if kinds is not None:
                        t["partner_link" if kinds[int(I[j])] == "link" else "partner_section"] += 1
    return t, (dups, pts)'''
assert old in s; s = s.replace(old, new)
s = s.replace('conn_points=0, arc_4_8=0, arc_8_16=0, arc_16p=0, partner_section=0, partner_link=0)\n    dups', 'conn_points=0, arc_4_8=0, arc_8_16=0, arc_16p=0, partner_section=0, partner_link=0, conn_dup_with_c_partner=0)\n    dups', 1)
open(p, "w", encoding="utf-8").write(s)
print("ok")
