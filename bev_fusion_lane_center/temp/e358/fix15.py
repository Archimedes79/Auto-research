import sys
sys.path.insert(0, "/mnt/c/Auto-research/bev_fusion_lane_center/temp/e358")
from edit_helper import apply
apply("/mnt/c/Auto-research/bev_fusion_lane_center/evaluation/jwire_check.py", [
("""            pput(card(**prim), ptwin)
            assert pt() == (True, []), pt()
            try:
                twin_check("rx", 0, "twin")                                                              # only the primary names exist now: no fall back to the snapped names, no mix-up of the two read-outs
                raise AssertionError("the snapped read-out must not find the primary cards")
            except FileNotFoundError:
                pass
            pput(card(), ptwin) """, """            try:
                pt()                                                                                     # only the snapped names exist so far: the primary read-out does not fall back to them
                raise AssertionError("the primary read-out must not read the snapped cards")
            except FileNotFoundError:
                pass
            pput(card(**prim), ptwin)
            assert pt() == (True, []), pt()
            pput(card(), ptwin) """),
("""            assert pt()[1] == ["card attach", "card snap_options", "the stored card is not the primary read-out (attach None, snap_options None)"], pt()""", """            assert pt()[1] == ["the stored card is not the primary read-out (attach None, snap_options None)"], pt()"""),
("""            put(card(**prim), dict(twin, **prim))                                                          # a card with attach / snap_nogeom is not the snapped read-out either
            assert twin_check("rx", 0, "twin")[1] == ["card attach", "card snap_options", "the stored card is not the snap read-out (attach True, snap_options ['snap_nogeom'])", "the twin card is not the snap read-out (attach True, snap_options ['snap_nogeom'])"], twin_check("rx", 0, "twin")""", """            put(card(**prim), dict(twin, **prim), counts(), counts())                                      # a card with attach / snap_nogeom is not the snapped read-out either
            assert twin_check("rx", 0, "twin")[1] == ["the stored card is not the snap read-out (attach True, snap_options ['snap_nogeom'])", "the twin card is not the snap read-out (attach True, snap_options ['snap_nogeom'])"], twin_check("rx", 0, "twin")"""),
])
print("ok")
