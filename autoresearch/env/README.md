# env - the python environment `gcar`

`environment_gcar.yml` is the conda export of the environment every problem runs in (`problem.json` `"python": "gcar"`): Python 3.12,
torch 2.8.0 + CUDA 12.9, every package with its version, no build strings and no machine path. It is the state of the PC the results
were made on.

## A new machine
From this folder, on Linux or WSL2 with Miniconda. Two packages come from their own indexes (the `+cu129` wheels of torch / torchvision and
`torch-scatter`), which the export does not record:
```
PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129 PIP_FIND_LINKS=https://data.pyg.org/whl/torch-2.8.0+cu129.html \
  conda env create -f environment_gcar.yml
python3 -m research machine init --data <dir> --runs <dir> --python gcar=$HOME/miniconda3/envs/gcar/bin/python --gpu-slots <n>   # from ../
```
Then the tests of `../README.md`. This recipe was written from the export and has not been run on a second machine; the environment of
this PC was built step by step (the script is `bev_fusion_lane_center/tools/env/setup/setup_gcar_env.sh` in the git tag `v1`).

Not in the export: the Gaussian rasteriser of the GaussianCaR foundation (a local build from its code). Only the extraction of new
foundation features needs it; the code in this repository reads the cached features.

## After a change of the environment
```
conda env export -n gcar --no-builds | grep -v "^prefix:" > environment_gcar.yml
```
and commit it: a result is repeatable only with the environment it was made in.
