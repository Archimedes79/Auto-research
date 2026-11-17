import json,glob,os
for f in sorted(glob.glob(runs/e359_p2_s*/tok_card_thr0.5_sec0.json)):
    m=json.load(open(f))[metrics]
    print(f.split(/)[1], topoF1
