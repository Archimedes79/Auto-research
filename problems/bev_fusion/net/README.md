# net - the model: preprocessing, network, losses

The model of the camera + radar BEV system, apart from the training (`../training/`) and the evaluation (`../evaluation/`). Three layers, each
importing only itself and the layers below it: **prep** < **model** < **loss**; on top of them `chain.py`, the one entry both the trainer and
the evaluation read the network through. Nothing here imports `training/`, `evaluation/` or the training tooling (`trainkit`).
`../tests/test_net_layering.py` checks this for every import (a function's and a main block's too) and names the two exceptions: the offline
feature-cache builders of prep run the frozen camera heads, and the student encoder reads a trained student with the checkpoint loader of
`trainkit`.

| where | what |
|---|---|
| `netpath.py` | the folder the code works in (`SITE` = `<data root>/bev_fusion`: caches, pretrained weights, runs) |
| `prep/` | before the first learned weight: datasets and batches, geometry, image transforms, augmentation, feature and image caches, visibility maps, targets, splits |
| `model/` | the network: camera and radar encoders, temporal and camera-radar fusion, the BEV trunk, the time mover and the heads, the input / output contract (`interface.py`); the whole chain is `model.py` |
| `loss/` | the losses and the loss weights (`losses.py`, `object_loss.py`, `motion_loss.py`, `radar_loss.py`, `loss_extras.py`), the terms only a training step has (`train_terms.py`: cross-time consistency, radar side loss, student anchor) and the student's loss against the teachers (`student_loss.py`) |
| `chain.py` | a run's network: `ChainConfig` (the config fields the data, the network and the loss read), `read_config`, `build`, `load_weights` (a checkpoint's EMA weights), `forward` and `forward_loss`; `REMOVED` = the options that left the code and how a stored config that names them is read |

Modules are imported absolutely by their package name (`from bev_fusion.net.model.model import ModelConfig`; the style: `../tests/test_net_layering.py`).

**What a file does is its module docstring.** The chain and the rules it obeys: `../DESIGN.md`. Every module's test is
`../tests/test_<module>.py` (`python3 -m research tests bev_fusion` runs them all; `../README.md` says what each needs).
