# Phase 41 — Cross-Validation Stability Analysis

**Version:** 1.1 | **Tier:** Micro | **Date:** 2026-03-26

## Goal
Analyze reproducibility of RF regression across multiple random seeds and fold counts.
Show that LOO-CV is stable but K-fold varies with seed on small datasets.

CLI: `python main.py --input data/compounds.csv`

Outputs: cv_stability.csv, cv_stability.png

## Logic
- Load compounds.csv, compute ECFP4 fingerprints (radius=2, nBits=2048, useChirality=True)
- LOO-CV: single deterministic pass, collect R² (gold standard baseline)
- K-fold CV: for K in [3, 5, 10], repeat with 20 random seeds (shuffle=True, random_state=seed)
- Each repeat: train RF (n_estimators=200, random_state=42), predict held-out fold, compute R²
- Collect per-K distributions: mean, std, min, max across 20 seeds
- Plot: scatter of R² per seed per K, diamond at mean, LOO-CV horizontal reference line

## Key Concepts
- LOO-CV determinism vs K-fold seed sensitivity on small datasets
- scikit-learn `KFold` with shuffle and varying random_state
- Variance-bias tradeoff: fewer folds = more test variance, more folds = closer to LOO
- RF `RandomForestRegressor` (n_estimators=200) with ECFP4 fingerprints

## Verification Checklist
- [x] LOO-CV R² = 0.729 (deterministic, same as Phase 37)
- [x] 3-fold variance (±0.064) > 5-fold (±0.031) > 10-fold (±0.020)
- [x] K-fold means converge toward LOO-CV as K increases
- [x] cv_stability.csv and cv_stability.png saved to output/

## Risks
- 20 random seeds may not fully characterize the K-fold distribution for K=3 (high variance)
- RF random_state is fixed at 42 for all folds; varying it too would add another variance source

## Actual Results (v1.1)

| Method | R² | Variance |
|---|---|---|
| LOO-CV | 0.7288 | deterministic |
| 3-fold (20 seeds) | 0.667 ± 0.064 | [0.528, 0.762] |
| 5-fold (20 seeds) | 0.703 ± 0.031 | [0.636, 0.752] |
| 10-fold (20 seeds) | 0.715 ± 0.020 | [0.681, 0.752] |

**Key insight:** LOO-CV is the most stable and closest to true leave-one-out performance. As K increases, variance decreases and estimates converge toward LOO. 3-fold has high variance (±0.064) — unreliable for small datasets. LOO-CV is the gold standard for n=45.
