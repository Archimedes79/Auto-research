# E094_trace-network-stem-pyramid-seed-head

**Trace network (1.67 M): stem pyramid, seed head, fan sampling, covered crop, two transformer blocks, merge pointer**

| | |
|---|---|
| experiment id | E94 |
| theme | T5 - Learned graph extractor and registration-free navigation head (system B) (loops 10 - 13) |
| tag | E - E2E-usable: learned or parameter-free, differentiable (the trace network has discrete bookkeeping inside) |
| sparsification stage | SP1 seeds and key points (peak picking, node heat maps); SP5 learned sparse extractor (step choice, vertices, events) |
| building blocks | BB02 deformable convolution / attention; BB03 grid sampling, spatial transformer, warping; BB05 peak picking: NMS, top-k, Gumbel top-k; BB11 self-attention; BB14 pointer attention; BB18 position codes; BB19 output heads: class + residual, splines; BB20 generative / sequential heads |
| loop | loop10 |

## Result

F1 0.883 / 0.851 (VAL / TEST) against 0.891 / 0.867 for the rule graph; no hand-built tracer, assembler or registration; seeds are heat-map peaks, vertices are appended by bookkeeping

## Techniques used (register K ids)

- K16: learned trace network: fan sampling, argmax steps, ring wavefront  (internal)
- K14: iterative graph growth guided by a CNN decision  (checked (abstract))
- K84: graph grown vertex by vertex with a transformer, imitation with corrective expert actions  (checked loops 8, 10)
- K48: nodes + pairwise edge classifier  (checked)

Sources and paper notes: [../../literature/register.md](../../literature/register.md)

## Data

- `results/`: 65 small files from `e94_a`, `e94_a_evb.log`, `e94_a_final_or.log`, `e94_a_ko`, `e94_a_ko.log`, `e94_a_oracle`, `e94_a_order`, `e94_a_order.log` ...
- raw outputs (arrays, checkpoints): `../../runs/` where still kept, otherwise staged in `../../_to_delete/runs/`
