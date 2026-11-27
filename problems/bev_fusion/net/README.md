# net - the model: preprocessing, network, losses

The model of the camera + radar BEV system, apart from the training (`../training/`). Three layers, each importing only itself and the layers
below it: **prep** < **model** < **loss**; `training/` stands above all three, and nothing here imports it or its tooling (`trainkit`).
`../tests/test_net_layering.py` checks this and names the exceptions (the offline feature-cache builders of prep run the frozen camera heads,
one forward of `prep/feature_cache.py` into `model/cached_model.py`, and the checkpoint loader of `trainkit` in the student encoder).

| folder | what |
|---|---|
| `prep/` | before the first learned weight: datasets and batches, geometry, image transforms, augmentation, feature and image caches, visibility maps, targets, splits |
| `model/` | the network: camera and radar encoders, temporal and camera-radar fusion, the BEV trunk, the time mover and the heads, the input / output contract (`interface.py`); the whole chain is `model.py` |
| `loss/` | the losses and the loss weights |

Modules are imported flat: `netpath.py` puts the three folders on the path (module names are unique), and names the folder the code works in
(`SITE`).

**What a file does is its module docstring.** The chain and the rules it obeys: `../DESIGN.md`. Many modules carry a self-test:
`python net/<layer>/<module>.py` from `problems/bev_fusion/` with the module path of `problem.json` `pythonpath` on `PYTHONPATH` (some need
nuScenes mini or the pretrained weights). The tests of the layers together: `../tests/`.
