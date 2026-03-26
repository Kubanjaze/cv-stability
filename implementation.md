# Phase 41 — Cross-Validation Stability Analysis

**Version:** 1.1 | **Tier:** Micro | **Date:** 2026-03-26

## Goal
Analyze reproducibility of RF regression across multiple random seeds and fold counts.
Show that LOO-CV is stable but K-fold varies with seed on small datasets.

CLI: `python main.py --input data/compounds.csv`

Outputs: cv_stability.csv, cv_stability.png

## Actual Results (v1.1)

| Method | R² | Variance |
|---|---|---|
| LOO-CV | 0.7288 | deterministic |
| 3-fold (20 seeds) | 0.667 ± 0.064 | [0.528, 0.762] |
| 5-fold (20 seeds) | 0.703 ± 0.031 | [0.636, 0.752] |
| 10-fold (20 seeds) | 0.715 ± 0.020 | [0.681, 0.752] |

**Key insight:** LOO-CV is the most stable and closest to true leave-one-out performance. As K increases, variance decreases and estimates converge toward LOO. 3-fold has high variance (±0.064) — unreliable for small datasets. LOO-CV is the gold standard for n=45.
