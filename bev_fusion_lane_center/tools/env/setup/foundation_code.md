# Foundation code: where it comes from (recipe for `~/work`)

Third-party code is not in git (D11): it is cloned again from these remotes at these commits. Licences are as read on the repositories; GPL / unlicensed code stays read-only. Copied on 2026-10-25 from the staged literature file `_to_delete/literature_lean_2026-10-25/repos/README.md`.

```
cd ~/work && git clone <remote> <folder> && git -C <folder> checkout <commit>
```

GaussianCaR weights (`~/work/gaussiancar_ckpt`), recorded 2026-11-24 from the local copy: file `nuscenes_map.ckpt`, 1,130,353,854 bytes, modified 2026-10-13 19:49, sha256 `a184b022ef6d73fcf3abd075176273912a0e1fbbdf589e284151f1e388fb9baa`. Source named by the repository README: the Hugging Face repository `santimontieleu/gaussiancar` (read in `bev_fusion/literature/code/gaussiancar_fusion.md`, 2026-10-25). Still open: the exact download URL and who downloaded it are not recorded, so it is not proven that this file is the one on Hugging Face; compare the sha256 with the file there when it is fetched again. The vehicle checkpoint (`nuscenes_vehicle.ckpt`) is not on this machine.

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
