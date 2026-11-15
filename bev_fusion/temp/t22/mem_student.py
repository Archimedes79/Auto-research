"""scratch: peak GPU memory and time of 9 student forwards kept in one graph + one backward (a stage-B micro-batch: batch 3 x clip of 3), with and without checkpoint_encode"""
import sys, time
import torch
sys.path.insert(0, "training"); sys.path.insert(0, ".")
from student_encoder import StudentEncoder

for kind in ("efficientvit_b1", "resnet18"):
    for flag in (False, True):
        torch.manual_seed(0)
        st = StudentEncoder(kind, pretrained=False).cuda()
        st.keep_bn_eval(); st.checkpoint_encode = flag; st.train()
        imgs = [torch.randint(0, 255, (1, 3, 256, 704), dtype=torch.uint8, device="cuda") for _ in range(9)]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base = torch.cuda.memory_allocated()
        t = time.time()
        outs = [st(i) for i in imgs]
        loss = sum(o["logit16"].square().mean() + o["ctx8"].square().mean() for o in outs)
        torch.cuda.synchronize(); fwd_peak = torch.cuda.max_memory_allocated() - base
        loss.backward(); torch.cuda.synchronize()
        print(f"{kind:16s} checkpoint_encode={flag!s:5s} after the 9 forwards {fwd_peak / 1e9:5.2f} GB, peak with backward {(torch.cuda.max_memory_allocated() - base) / 1e9:5.2f} GB, {time.time() - t:5.2f} s", flush=True)
        del st, imgs, outs, loss; torch.cuda.empty_cache()
