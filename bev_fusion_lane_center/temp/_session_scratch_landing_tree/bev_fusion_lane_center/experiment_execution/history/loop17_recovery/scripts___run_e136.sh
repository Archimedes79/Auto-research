#!/bin/bash
# E136: joint end-to-end fine-tune (builder + route head along the edges, route loss through the node tokens), three arms, then route evaluation on failure series
S=/mnt/c/ML-Experiment/scripts
L=/mnt/c/ML-Experiment/out/loop2/e17/logs
bash $S/_py14.sh p17_joint.py --name e17_joint_a --iters 600 --bs 4 --lam 1.0 > $L/e136_a.log 2>&1
bash $S/_py14.sh p17_joint.py --name e17_joint_b --iters 600 --bs 4 --lam 0.05 > $L/e136_b.log 2>&1
bash $S/_py14.sh p17_joint.py --name e17_joint_c --iters 600 --bs 4 --lam 0.05 --no_builder_grad > $L/e136_c.log 2>&1
SER=dark_single,dark_maxdecay_0.95,blur_maxdecay_0.95
SET="M=192 K0=96"
bash $S/_py14.sh p17_route.py --out e136_route --set $SET --prev 0 --series $SER --routes p16_f2hr_route_s0 --sigmas 0 --tag_suffix _base > $L/e136_eval_base.log 2>&1
for arm in a b c; do
  bash $S/_py14.sh p17_route.py --out e136_route --set $SET --prev 0 --ckpt e17_joint_$arm --series $SER --routes e17_joint_$arm --sigmas 0 --tag_suffix _$arm > $L/e136_eval_$arm.log 2>&1
done
echo finished > $L/e136.done
