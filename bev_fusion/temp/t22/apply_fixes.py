import re
def sub(path, old, new, count=1):
    s = open(path, encoding="utf-8").read()
    assert s.count(old) == count, (path, old[:60], s.count(old))
    open(path, "w", encoding="utf-8").write(s.replace(old, new))

# 1. feature_cache: the key made once in the forward; honest comment
sub("training/feature_cache.py",
    "                    if recompute_lift:      # t22: the student's depth + context carry a graph into the lift; without this the 118 x C frustum features of the 9 frames of a micro-batch stay in memory\n"
    "                        vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))\n",
    "                    if recompute_lift:      # t22: the student's depth + context carry a graph into the lift; without this the 118 x C frustum features of the 9 frames of a micro-batch stay in memory\n"
    "                        if key is None:     # the key the lift would make from the GPU tensors (three waits for the GPU): made once here, so that no recomputation inside a backward has to wait\n"
    "                            key = lift_key(*args[2:])\n"
    "                        vols.append(collapse(torch.utils.checkpoint.checkpoint(lift, *args, key=key, use_reentrant=False)))\n")
sub("training/feature_cache.py",
    "try:\n    from .camera_encoder import FrontCameraEncoder, camera_vector\n",
    "try:\n    from .lift_pd import lift_key\n    from .camera_encoder import FrontCameraEncoder, camera_vector\n")
open("temp/t22/_fc_check.txt", "w").write("ok")

# 2. student_encoder: BatchNorm guard + comment
sub("training/student_encoder.py",
    "        # t22 stage B: 9 key frames of a micro-batch (batch 3 x clips of 3, the graph kept over the clip) held the backbone activations of every frame at once: peak 16.3 GB on the 12 GB card\n"
    "        # (smoke run 2026-11-05, 56 s per step from GPU-memory paging). The encoder is then recomputed in the backward (+1 forward of the backbone, about 5 ms per frame) and only its four\n"
    "        # output maps (2.7 MB per frame) stay. The BatchNorm of a student in the chain is in eval mode and the backbone has no dropout, so the recomputation is the same function.\n"
    "        if self.checkpoint_encode and self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters()):\n",
    "        # t22 stage B: 9 key frames of a micro-batch (batch 3 x clips of 3, the graph kept over the clip) held the backbone activations of every frame at once: peak 16.3 GB on the 12 GB card\n"
    "        # (smoke run 2026-11-05, 56 s per step from GPU-memory paging). The encoder is then recomputed in the backward and only its four output maps (2.7 MB per frame) stay. Every backward\n"
    "        # that reaches a frame recomputes it (with seq_bptt a clip of 3 runs the encoder 3 + 1 + 2 + 3 = 9 times, not 6). Only with BatchNorm held in eval mode (keep_bn_eval): a train-mode\n"
    "        # BatchNorm would count its statistics twice; the backbone has no dropout, so the recomputation is the same function.\n"
    "        if self.checkpoint_encode and self._bn_eval and self.training and torch.is_grad_enabled() and any(p.requires_grad for p in self.backbone.parameters()):\n")

# 3. model.py comment
sub("training/model.py",
    "with seq_bptt the graphs of the student of every key frame of a clip stay in memory (not measured)",
    "with seq_bptt the graphs of every key frame of a clip stay in memory: measured 16.3 GB at batch 3 x clips of 3 (paging on a 12 GB card), 9.1 GB with the encoder and the lift recomputed in the backward (checkpoint_encode, camera_bevs; +2.6 s per step)")
print("ok")
