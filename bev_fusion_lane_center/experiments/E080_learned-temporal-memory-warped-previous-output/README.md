# E080_learned-temporal-memory-warped-previous-output

**Learned temporal memory: the warped previous output as extra input channels**

| | |
|---|---|
| experiment id | E80 |
| theme | T6 - Time and sensor failure (loops 5 - 13) |
| tag | D - dropped: tried, no gain or worse |
| sparsification stage | SP0 not part of the sparsification chain (foundation, map, time, planner, analysis) |
| building blocks | BB07 recurrent cells (GRU, ConvGRU); BB10 memory bank / query propagation |
| loop | loop07 |

## Result

+0.000 - 0.005

## Techniques used (register K ids)

- K45: temporal map propagation and stream-query denoising, memory  (checked)
- K78: global raster memory accumulated from past predictions  (checked loop 7 (html))
- K79: propagate confident queries; fast and slow latent memory  (checked loops 7, 14)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 22 small files from `e80_t_ema_free`, `e80_t_ema_full`, `e80_t_free`, `e80_t_free_thr`, `e80_t_full`, `e80_t_none`, `e80_test_t_ema_full`, `e80_val_t_ema_free` ...
- `figures/`: 1 images from the report sources
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
