import sys
import os
sys.argv = ["x"]
__file__ = os.path.abspath("tests/test_student_chain.py")
src = open("tests/test_student_chain.py").read()
# run the test up to the frozen block, then inspect
cut = src.index("# trainable: every tensor gets a gradient")
exec(compile(src[:cut], "tests/test_student_chain.py", "exec"))
for n, p in fro.named_parameters():
    if n.startswith(("cam_reduce", "cam_merge")):
        print(n, p.requires_grad, None if p.grad is None else float(p.grad.abs().sum()))
print(out.requires_grad, float(out.abs().sum()), torch.isfinite(out).all())
