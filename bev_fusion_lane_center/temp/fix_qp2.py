p = "training/grid/qpredict.py"
s = open(p, encoding="utf-8").read()
old = s[s.index("        if keep_maps:"): s.index("        preds.append(")]
new = '''        if keep_maps:
            P = out["P"]
            lines = P["lines05"][b].float()
            c1 = P["centre1"][b].float()
            maps = dict(lines05=torch.cat([lines[:3].sigmoid(), 2.0 * lines[3:].sigmoid()]).cpu().numpy(), centre1=torch.cat([c1[:1].sigmoid(), c1[1:5], c1[5:6].sigmoid()]).cpu().numpy(),
                        events1=c1[6:].sigmoid().cpu().numpy(), road2=P["road2"][b].float().sigmoid().cpu().numpy())            # probabilities (sigmoid applied), distances in metres
'''
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
p = "training/grid/qformat.py"
s = open(p, encoding="utf-8").read()
s = s.replace("  maps    None or dict of the dense layers for the pictures: lines05 (6, 200, 200), centre1 (6, 100, 100), events1 (6, 100, 100), road2 (2, 50, 50) (probabilities / logits as made)",
              "  maps    None or dict of the dense layers for the pictures, PROBABILITIES (sigmoid applied, never logits): lines05 (6, 200, 200) = 3 line probabilities + 3 truncated distances in metres, centre1 (6, 100, 100)\n            = centre probability, cos, sin, ox, oy, keep gate, events1 (6, 100, 100), road2 (2, 50, 50) = road, intersection")
open(p, "w", encoding="utf-8").write(s)
