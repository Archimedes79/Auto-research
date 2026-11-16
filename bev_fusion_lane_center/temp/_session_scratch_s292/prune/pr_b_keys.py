import torch
sd = torch.load(__import__('os').path.join(__import__('os').environ['BFLC_ROOT'], 'runs/e292_c2_s0/model.pt'), map_location='cpu')['state']
for k, v in sd.items():
    if k.split('.')[0] in ('nodes', 'links', 'menc', 'asg') and v.dim() >= 1 and not k.endswith('num_batches_tracked'):
        print(f"{k:55s} {tuple(v.shape)}")
