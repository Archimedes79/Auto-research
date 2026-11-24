# Foundation code: where it comes from (recipe for `~/work`)

Third-party code is not in git (D11): it is cloned again from these remotes at these commits. Licences are as read on the repositories; GPL / unlicensed code stays read-only. Copied on 2026-10-25 from the staged literature file `_to_delete/literature_lean_2026-10-25/repos/README.md`.

```
cd ~/work && git clone <remote> <folder> && git -C <folder> checkout <commit>
```

GaussianCaR weights (`~/work/gaussiancar_ckpt`), recorded 2026-11-24 from the local copy: file `nuscenes_map.ckpt`, 1,130,353,854 bytes, modified 2026-10-13 19:49, sha256 `a184b022ef6d73fcf3abd075176273912a0e1fbbdf589e284151f1e388fb9baa`. Source named by the repository README: the Hugging Face repository `santimontieleu/gaussiancar` (read in `bev_fusion/literature/code/gaussiancar_fusion.md`, 2026-10-25), download link of the file: https://huggingface.co/santimontieleu/gaussiancar/resolve/main/nuscenes_map.ckpt (licence Apache-2.0, not gated, repository revision `8e439c1085577efbdaf9ca31a3531d08901f16c6`, last modified 2026-02-09). Checked 2026-11-24 against the repository metadata (`https://huggingface.co/api/models/santimontieleu/gaussiancar/tree/main`, read only, nothing downloaded): the file there has the same size (1,130,353,854 bytes) and the same sha256 (`a184b022...`) as the local copy, so the local file is the one on Hugging Face. Who downloaded it and when is not recorded. The vehicle checkpoint (`nuscenes_vehicle.ckpt`, 1,094,842,742 bytes, sha256 `09d8b7710c29c20ad8aa1e380345fcd19146fac58af293d1ffe4341d0d3ea9c6` on Hugging Face) is not on this machine; it lies in the same repository (`.../resolve/main/nuscenes_vehicle.ckpt`) and a download needs the owner's word.

## Clones

### CRN (Camera Radar Net), ICCV 2023

- local path: `~/work/CRN` (present)
- remote: https://github.com/youngskkim/CRN.git
- last commit: 5e9d2fa 2023-12-07 Fix broken link
- licence file: LICENSE
- used for: Detection-pretrained camera-radar foundation of loops 1-3 (E15-E26); replaced by GaussianCaR from E27.

### GaussianCaR, ICRA 2026

- local path: `~/work/gaussiancar` (present)
- remote: https://github.com/santimontiel/gaussiancar
- last commit: c0e748b 2026-10-07 📝 docs: Update citation to ICRA 2026 proceedings
- licence file: LICENSE
- used for: Frozen map-pretrained camera-radar foundation of all later loops (39 channels, 200 x 200 cells of 0.5 m).

### BEVFusion (MIT HAN Lab)

- local path: `~/work/bevfusion` (present)
- remote: https://github.com/mit-han-lab/bevfusion.git
- last commit: 326653d 2024-07-30 [ROCm] add support for ROCm/HIP (#509)
- licence file: LICENSE
- used for: Camera-lidar strand E01-E14, results deleted 2026-10-13; code kept.

### gcar probe scripts

- local path: `~/work/gcar_probe` (present)
- licence file: none found at top level
- used for: First trial runs of the GaussianCaR environment (own scripts, not a clone).

Weights: `~/work/gaussiancar_ckpt/nuscenes_map.ckpt` (GaussianCaR map-segmentation checkpoint).
