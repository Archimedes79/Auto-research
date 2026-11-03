# Clean-up log

Written by `autoresearch/tools/package.py clean|gt --apply` (D40): what was deleted, when, and how to get it back.

- 2026-11-02T01:26:06: GT store b84e8ac9b2 (v7) is being deleted; regenerate from the tag and the options of its index.json (git show autogt/<version>)
- 2026-11-02T01:26:53: GT store 15efbde2d1 (v5, 34149 keyframes, options {}) deleted by hand at Mirko's word (chat 2026-11-02: 'lösche v5 dateien', and the experiments to v8) although `package.py gt` keeps it for the records of t10.dual_long_s0 (on hold, Mirko answered no to finishing it) and t8.ctl_s1 (finished: ckpt_00003000.pt and eval exist, the record still says running); no process was running, and nothing under bev_fusion/training or evaluation reads data/autogt (the runs read the trainer's own cache bev_fusion/data/cache/gt, which is untouched); the v8 store was compared with it frame by frame on 2026-11-02 first; regenerate from the tag autogt/v5 with the default options (git show autogt/v5)
