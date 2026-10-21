# SpectralGate-Net

Code for "SpectralGate-Net: Wavelet-Domain Reliability Gating for RGB-Thermal Camouflaged Object Segmentation under Simulated Thermal Crossover".

A simulated thermal (LWIR) channel is generated from COD10K, and an RGB-only model, early concatenation, and wavelet-gated fusion variants are compared as the target-background temperature difference delta goes to 0.

## Data
COD10K (camouflaged subset) is NOT included; download it from the official source (see Fan et al., CVPR 2020). It is released for non-commercial use. Expected folders: Train/Image, Train/GT_Object, Test/Image, Test/GT_Object.

## Run
    python sgn.py train --mode full   --root /path/to/COD10K-v3 --epochs 12
    python sgn.py train --mode fixed  --root /path/to/COD10K-v3
    python sgn.py train --mode concat --root /path/to/COD10K-v3
    python sgn.py train --mode rgb    --root /path/to/COD10K-v3
    python sgn.py train --mode full --lgate 0 --root /path/to/COD10K-v3
    python sgn.py train --mode full --lgate 0 --raw_desc --root /path/to/COD10K-v3
    python sgn.py report

Add `--seed 1` for a second seed. `--smoke` runs a one-minute check on random data.

## Results
`results/` contains the JSON results and logs from the runs reported in the paper (one Tesla T4 each on Kaggle). GPU runs are not guaranteed to be bit-for-bit reproducible.

## License
MIT for the code. COD10K has its own license.
