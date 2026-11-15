import torch
for v in (torch.arange(5.0), torch.arange(5), torch.arange(4, dtype=torch.long), torch.zeros(0), torch.randn(3, 2)):
    keep = torch.zeros(v.shape[0], dtype=torch.bool)
    a, b = v[keep], v[:0]
    assert a.shape == b.shape and a.dtype == b.dtype and a.device == b.device and a.stride() == b.stride() or a.numel() == 0, (a.shape, b.shape)
    print(tuple(a.shape), tuple(b.shape), a.dtype == b.dtype, a.is_contiguous(), b.is_contiguous())
# in-place on the view of an empty tensor cannot reach the parent
v = torch.arange(3.0); w = v[:0]; w.zero_(); assert v.tolist() == [0.0, 1.0, 2.0]
print("ok")
