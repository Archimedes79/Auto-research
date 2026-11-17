import json
for S in (0, 1):
    t = open(f"temp/e353/check24_s{S}.log").read()
    d = json.loads(t[t.index("{"):])
    b, j = d["b"], d["junction"]
    print(f"seed {S}: nodes matched {b['nodes_matched']} (only stored {b['nodes_only_in_ref']}, only new {b['nodes_only_in_new']}, max e of those {b['e_max_unmatched']:.3f}); max abs e {b['nodes']['e']:.1e} pos {b['nodes']['pos']:.1e} m dirv {b['nodes']['dirv']:.1e} flags {b['nodes']['flags']:.1e} s {b['nodes']['s']:.1e}; sections differ {b['sections_differ']} of {b['sections_compared']}; links matched {b['links_matched']} one-sided {b['links_only_in_ref']} + {b['links_only_in_new']} (their max p {b['p_max_one_sided']:.3f}); link p {b['links']['p']:.1e} a {b['links']['a']:.1e} b {b['links']['b']:.1e}")
    print(f"   junction: keypoints matched {j['keypoints_matched']} (only stored {j['keypoints_only_in_ref']}, only new {j['keypoints_only_in_new']}); kp_pos {j['max_abs']['kp_pos']:.1e} kp_dir {j['max_abs']['kp_dir']:.1e} kp_p {j['max_abs']['kp_p']:.1e} att_pos {j['max_abs']['att_pos']:.2f} att_dir {j['max_abs']['att_dir']:.2f} att_p {j['max_abs']['att_p']:.3f} logit {j['max_abs']['logit']:.1e} scale {j['max_abs']['scale']:.1e}; attached ends moved by more than 0.05 m: {j['att_moved']} of {j['keypoints_matched']} (max {j['att_moved_max_m']:.2f} m); frames with an unmatched keypoint {j['frames_with_unmatched']}")
