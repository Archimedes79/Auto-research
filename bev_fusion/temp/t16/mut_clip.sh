#!/bin/bash
# scratch: three mutants of the split clip must fail tests/test_clip_split.py
cd "$(dirname "$0")/../.." || exit 1
M=temp/t16/mut
rm -rf $M && mkdir -p $M/tests && cp -r training $M/training && rm -rf $M/training/__pycache__ && cp tests/test_clip_split.py $M/tests/
cd $M
run() { OMP_NUM_THREADS=2 /home/ml/miniconda3/envs/gcar/bin/python tests/test_clip_split.py 2>&1 | tail -1; }
echo "baseline: $(run)"
cp training/train.py /tmp/train_orig.py
sed -i 's/gn = clip_gradients(params, grad_clip, split)/gn = clip_gradients(params, grad_clip, ())/' training/train.py; echo "M5 optimizer_step ignores split: $(run)"; cp /tmp/train_orig.py training/train.py
sed -i 's/taken = {id(p) for g in split for p in g}/taken = {id(p) for p in split[0]}/' training/train.py; echo "M13 taken from the first group only: $(run)"; cp /tmp/train_orig.py training/train.py
sed -i 's/before_clip=pstep.grads if pstep is not None else None, split=clip_split)/before_clip=pstep.grads if pstep is not None else None)/' training/train.py; echo "M_call the train() call site drops split: $(run)"; cp /tmp/train_orig.py training/train.py
sed -i 's/for k in ("cudnn_benchmark", "tf32", "grad_clip_split") if/for k in ("cudnn_benchmark", "tf32") if/' training/train.py; echo "M_resume the resume hole is back: $(run)"; cp /tmp/train_orig.py training/train.py
cd ../../.. && rm -rf $M
