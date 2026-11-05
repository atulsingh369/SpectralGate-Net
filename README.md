# SpectralGate-Net (v2.0)

This repository contains the code, simulated thermal crossover protocol, and experiments for two related studies on RGB-Thermal camouflaged object segmentation.

**1. Bio-AI Conclave 2026**  
*Mitigating Modality Over-Reliance at Thermal Crossover: A Controlled RGB-T Study for Camouflaged Animal Segmentation*  
Investigates the structural failure of early concatenation at thermal crossover and maps a Pareto-optimal trade-off frontier using Modality Dropout and Oracle gating.

**2. NICEDT 2027**  
*SpectralGate-Net: Wavelet-Domain Reliability Gating for RGB-Thermal Camouflaged Object Segmentation under Simulated Thermal Crossover*  
Proposes a wavelet-domain architecture (Haar DWT + B-HFCA) and a differentiable dynamic reliability gate, establishing negative results for scalar gating.

## Data Requirement
The base visible-spectrum images are drawn from the camouflaged subset of **COD10K** (Fan et al., CVPR 2020). COD10K is NOT included here; download it from the official source (released for non-commercial use).  
Expected folder structure:
* `COD10K-v3/Train/Image`
* `COD10K-v3/Train/GT_Object`
* `COD10K-v3/Test/Image`
* `COD10K-v3/Test/GT_Object`

The LWIR (thermal) channel is synthesized dynamically during training and evaluation based on physical Planck radiance and a parametrized target-background contrast ($\delta$).

---

## Running SpectralGate-Net v2.0: Modality Dropout & Oracle Gates (Bio-AI Paper)
The `phase1.py` script executes the Modality Dropout sweeps ($p=0.10, 0.25, 0.50$), supervised gating, the Oracle gate ceiling, and the threshold routing rule.

**Train Models:**
```bash
# Run baseline, standard concat, and modality dropout sweeps across 3 seeds
python phase1.py --configs rgb_only,concat,concat_drop,concat_drop_d0,concat_d0_p10,concat_d0_p50 --seeds 0,1,2 --root /path/to/COD10K-v3

# Run gating and oracle bounds across 3 seeds
python phase1.py --configs gate_raw,gate_sup,gate_sup_drop,gate_oracle --seeds 0,1,2 --root /path/to/COD10K-v3
```

**Evaluate Threshold Rule & Generate Report:**
```bash
# Computes the empirical upper-bound switch and the threshold heuristic
python phase1.py --rule --seeds 0,1,2 --root /path/to/COD10K-v3

# Generates LaTeX tables and PDF plots (miou_vs_delta.png, alpha_vs_delta.png)
python phase1.py --report
```
*Use `--smoke` for a 1-minute CPU/GPU sanity check with synthetic data.*

---

## Running SpectralGate-Net v1.0: Wavelet Architecture (NCEDT Paper)
The `sgn.py` script executes the Haar-wavelet decomposition and D-DRG gate experiments.

**Train Models:**
```bash
python sgn.py train --mode full   --root /path/to/COD10K-v3 --epochs 12
python sgn.py train --mode fixed  --root /path/to/COD10K-v3
python sgn.py train --mode concat --root /path/to/COD10K-v3
python sgn.py train --mode rgb    --root /path/to/COD10K-v3
python sgn.py train --mode full --lgate 0 --root /path/to/COD10K-v3
python sgn.py train --mode full --lgate 0 --raw_desc --root /path/to/COD10K-v3
```

**Generate Report:**
```bash
python sgn.py report
```

---

## Results & Reproducibility
* `results.jsonl` and `phase1/` directories contain the raw logging metrics from our multi-seed Tesla T4 (Kaggle) runs. 
* Due to non-deterministic GPU atomic operations in PyTorch, independent GPU runs are mathematically sound but may not be bit-for-bit identical to the provided JSON logs.

## License
The code in this repository is released under the **MIT License**.  
The COD10K dataset retains its original license governing non-commercial use.