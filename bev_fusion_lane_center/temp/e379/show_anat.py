import json, sys
for f in sys.argv[1:]:
    d = json.load(open(f)); c = d["categories"]
    print(f.split("/")[1], "delivered", d["delivered"], "tp %d (%.3f)" % (c["tp"]["n"], c["tp"]["share"]), "dup %.3f" % c["dup_of_true"]["share"], "wrong %.3f" % c["wrong_pair"]["share"], "false_end %.3f" % c["false_end"]["share"],
          "tp/GT %.3f  ap_long %.3f  ap_exist %.3f  recall_ceiling %.3f" % (d["tp_over_gt_long"], d["ap_long"], d["ap_existence"], d["recall_ceiling"]))
