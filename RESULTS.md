# Results Log — Strong Lens Inference

All valuable numbers from every pipeline run, so nothing lives only in a terminal
scrollback. Newest runs at the bottom. Update this file after each
`generate` / `train` / `evaluate` cycle.

---

## Fixed setup (all runs below, unless noted)

| Setting | Value |
|---|---|
| Image grid | 64×64 px, 0.05"/px → 3.20" field of view |
| PSF | Gaussian, FWHM 0.1" |
| Noise | background_rms = 0.1, exp_time = 100.0 |
| Source amplitude | SOURCE_AMP = 20.0 (Runs 1-3) → **150.0 (Runs 4-5 onward)** |
| Dataset | 8000/1000/300 (Runs 1-3) → **20000/2000/300 (Runs 4-5 onward)** |
| Network | SUMMARY_DIM = 48, COUPLING_DEPTH = 4 |
| Training | 40 epochs, batch 32, lr 1e-3, seed 42 |

Priors: theta_E (0.7, 1.6) | q (0.6, 1.0) | phi (0, pi) | gamma_ext (0, 0.08) |
phi_ext (0, pi) | x_s, y_s (-0.3, 0.3) | R_s (0.05, 0.25) | kappa (0, 0.2)

---

## Run 1 — 8 parameters, CPU (2026-07-11, morning)

**Hardware:** CPU only (torch backend), 11 simulation workers.
**Training time:** 1.36 hours (~125 s/epoch).

### Dataset SNR report
```
peak SNR: min=0.8  median=3.7  max=7.5     (target ~10-30 — NOT met)
```

### Training loss (negative log-posterior density)
| Epoch | train | val |
|---|---|---|
| 1 | 10.67 | 1472.31 |
| 5 | 6.61 | 7.36 |
| 10 | 4.79 | 5.82 |
| 15 | 3.47 | 4.81 |
| 20 | 2.38 | 5.06 |
| 22 | 1.96 | 4.32 |
| 25 | 1.40 | **4.29** (best val) |
| 30 | 0.84 | 5.20 |
| 35 | 0.22 | 5.58 |
| 40 | 0.10 | 5.65 |

Val loss plateaus ~epoch 17-25 while train keeps dropping → overfitting in late
epochs. Early stopping ~epoch 20-25 recommended.

### Evaluation
Figures saved: recovery.png, calibration.png, posterior_predictive.png
(2000 posterior samples per test system, sampling took ~9 s).

---

## Run 2 — 9 parameters (with kappa), CPU (2026-07-11, morning)

Same config, `SLI_INCLUDE_KAPPA=1`. Files carry `_kappa` suffix.

### Dataset SNR report
```
peak SNR: min=0.6  median=3.9  max=7.7
```

### Training loss
| Epoch | train | val |
|---|---|---|
| 1 | 12.11 | 392.22 |
| 10 | 6.24 | 6.45 |
| 14 | 5.16 | 6.02 |
| 22 | 3.47 | **5.70** (best val) |
| 30 | 2.40 | 6.83 |
| 40 | 1.68 | 6.62 |

### Mass-sheet degeneracy analysis (the key result)
```
kappa prior std           : 0.058
kappa mean posterior std  : 0.051
contraction               : +0.12   (near 0 = kappa barely constrained, as expected)
mean corr(kappa, theta_E) : -0.83   (strong trade-off = the degeneracy signature)
```

---

## Run 3 — 8 parameters, GPU (2026-07-11, evening)

**Hardware:** NVIDIA GeForce RTX 3060 (12 GB), torch 2.13.0+cu130, CUDA detected.
**Pipeline timing:** generate ~19:26, model saved 19:46, evaluate 19:48
→ **whole train step < 20 min** (vs 1.36 h on CPU, ~4-5× speedup).
Same config and seed as Run 1.

### Recovery R² (posterior mean vs truth, 300 test systems)
| Parameter | R² | Assessment |
|---|---|---|
| R_s | 0.95 | excellent |
| theta_E | 0.88 | very good |
| y_s | 0.84 | very good |
| x_s | 0.83 | very good |
| e2 | 0.49 | moderate |
| e1 | 0.47 | moderate |
| gamma1 | 0.21 | poor |
| gamma2 | 0.06 | ~unconstrained |

### Calibration (SBC rank histograms)
- gamma1 (and mostly e2): flat within band → calibrated (prior-dominated).
- x_s, y_s, R_s, theta_E: clear U-shape → posteriors **overconfident** (too narrow).

### Posterior predictive
Posterior draws reproduce ring radius, knot positions, and azimuthal light
distribution of the observed image. Residual variation mainly in knot brightness.

**Note:** per-epoch losses of this GPU run were terminal-only and not captured.

---

## SOURCE_AMP tuning sweep (2026-07-11, evening)

Peak SNR over 60 prior draws vs source amplitude (to meet the assignment's
10-30 target):

| SOURCE_AMP | min | median | max |
|---|---|---|---|
| 20 (current) | 1.0 | 3.8 | 5.5 |
| 60 | 2.7 | 8.9 | 12.1 |
| 100 | 4.1 | 12.6 | 16.7 |
| **150** | 5.7 | **16.3** | 21.3 |
| 200 | 7.1 | 19.4 | 25.1 |

**Recommendation:** SOURCE_AMP = 150 (or 200) puts typical images in the target range.

---

## Run 4 — 8 parameters, GPU, corrected SNR + 20k dataset (2026-07-11, night)

**Changes vs Run 3:** SOURCE_AMP 20 → 150 (tuned via the sweep below),
N_TRAIN 8000 → 20000, N_VAL 1000 → 2000, IMAGE_SCALE 0.5 → 4.0.
**Hardware:** RTX 3060 GPU. Training: **49.8 min** (117 ms/step; CPU was ~500 ms/step).

### Dataset SNR report — target met
```
peak SNR: min=4.6  median=16.1  max=26.9    (target ~10-30 ✓)
```

### Training loss (negative log-posterior density)
| Epoch | train | val |
|---|---|---|
| 1 | 9.92 | 39.37 |
| 5 | 2.80 | 1.76 |
| 10 | −0.02 | 0.55 |
| 15 | −1.51 | −1.19 |
| 20 | −2.36 | −3.39 |
| 25 | −3.29 | −4.28 |
| 30 | −3.91 | −5.33 |
| 35 | −4.39 | −5.78 |
| 40 | −4.50 | **−5.87** (best = final) |

**No overfitting** — val loss decreases monotonically through all 40 epochs and
stays *below* train loss. The 2.5× larger dataset fixed Run 1's overfitting.

### Recovery R² (300 test systems; Run 3 low-SNR values for comparison)
| Parameter | R² (Run 4) | R² (Run 3) |
|---|---|---|
| theta_E | **1.00** | 0.88 |
| R_s | **1.00** | 0.95 |
| x_s | **0.99** | 0.83 |
| y_s | **0.99** | 0.84 |
| e1 | **0.89** | 0.47 |
| e2 | **0.84** | 0.49 |
| gamma1 | **0.80** | 0.21 |
| gamma2 | **0.65** | 0.06 |

### Calibration (SBC)
U-shapes (overconfidence) are gone. e1/e2/gamma1/gamma2: flat = calibrated.
theta_E/x_s/y_s (mildly R_s): gentle central hump = mildly UNDERconfident
(slightly-too-wide posteriors — the safe direction).

### Posterior predictive
Posterior draws nearly indistinguishable from the observed image (ring radius,
all three knots, azimuthal distribution).

**Logs:** scratchpad `logs/{gen8,train8,eval8}.log`. Loss curves now auto-saved
by `main.py train` → `figures/training_loss.png`.

---

## Run 5 — 9 parameters (kappa), GPU, corrected SNR + 20k dataset (2026-07-11, night)

Same config as Run 4, `SLI_INCLUDE_KAPPA=1`. Training: **48.2 min**.

### Dataset SNR report
```
peak SNR: min=3.6  median=16.6  max=27.4
```

### Training loss
| Epoch | train | val |
|---|---|---|
| 1 | 11.00 | 25.52 |
| 10 | 1.59 | 1.34 |
| 20 | −1.04 | −2.17 |
| 30 | −2.55 | −3.97 |
| 40 | −3.18 | **−4.63** (best ≈ final) |

### Mass-sheet degeneracy analysis (the key result)
```
kappa prior std           : 0.058
kappa mean posterior std  : 0.048
contraction               : +0.17   (kappa still ~unconstrained, as theory demands)
mean corr(kappa, theta_E) : -0.93   (stronger than Run 2's -0.83 — higher SNR
                                     collapses the posterior onto the degeneracy ridge)
```

### Recovery R² with kappa (vs Run 4 without kappa)
| Parameter | with κ | without κ |
|---|---|---|
| theta_E | **0.93** | 1.00 ← degeneracy feeds uncertainty into θ_E |
| kappa | **0.18** | — (posterior ≈ prior, error bars span the prior) |
| e1 | 0.88 | 0.89 |
| e2 | 0.85 | 0.84 |
| gamma1 | 0.78 | 0.80 |
| gamma2 | 0.74 | 0.65 |
| x_s | 0.99 | 0.99 |
| y_s | 0.99 | 0.99 |
| R_s | 0.98 | 1.00 |

Only theta_E is affected by adding kappa — the degeneracy is specific to the
(kappa, theta_E) pair, exactly as the lensing equations predict.

**Figures:** `figures/{recovery,calibration,posterior_predictive}_kappa.png`,
`figures/training_loss_kappa.png`. Logs: scratchpad `logs/{genk,traink,evalk}.log`.

---

## Run 6 — 8 parameters, GPU, 50k dataset (2026-07-18, overnight)

**Changes vs Run 4:** N_TRAIN 20000 → 50000. Early stopping added (patience 10,
never triggered — best epoch = 40/40). Diagnostics switched to the
BayesFlow-native reference set: recovery (median + MAD + Pearson r), SBC
rank-ECDF difference with 95% bands, z-score vs contraction, prior pairplot,
fiducial-system posterior-vs-prior corner plot.
**Hardware:** RTX 3060. Generation: 50k+2k+300 in ~1 min (15 workers).
Training: **123 min** (118 ms/step), 40 epochs.

### Dataset SNR report
```
peak SNR: min=4.6  median=16.1  max=27.4    (target ~10-30 ✓)
```

### Training loss
Epoch 1: train 8.86 / val 6.83 → Epoch 40: train −6.19 / **val −8.05** (best =
final; monotone decrease, no overfitting; 20k run ended at val −5.87).

### Recovery (300 test systems; vs Run 4 at 20k)
| Parameter | r (50k) | R² (50k) | R² (20k) |
|---|---|---|---|
| theta_E | 0.999 | 1.00 | 1.00 |
| x_s | 0.999 | 1.00 | 0.99 |
| y_s | 0.999 | 1.00 | 0.99 |
| R_s | 0.999 | 1.00 | 1.00 |
| e1 | 0.970 | 0.94 | 0.89 |
| e2 | 0.960 | 0.92 | 0.84 |
| gamma1 | 0.928 | 0.86 | 0.80 |
| gamma2 | 0.895 | 0.80 | 0.65 |

### Calibration (SBC rank-ECDF difference, 95% simultaneous bands)
e1/e2/gamma1/gamma2: inside the band = calibrated. theta_E/x_s/y_s/R_s: symmetric
S-shape exiting the band (central ranks over-represented) = **mild
underconfidence** (conservative posteriors). Same shape at 20k and 50k → it is a
network-capacity effect, NOT a data-volume effect. Next knob if needed:
SUMMARY_DIM / COUPLING_DEPTH, not N_TRAIN.

### Fiducial inference (theta_E=1.15, q~0.85 lens, weak shear)
All truths within ~1.5 posterior std; theta_E posterior 1.152 ± 0.008.

---

## Run 7 — 9 parameters (kappa), GPU, 50k dataset (2026-07-18, overnight)

Same config as Run 6, `SLI_INCLUDE_KAPPA=1`. Training: **123 min**, 40 epochs,
final val loss **−6.79**. SNR: min=2.3 median=16.6 max=27.5.

### Mass-sheet degeneracy analysis (the key result)
```
kappa prior std           : 0.058
kappa mean posterior std  : 0.048
contraction               : +0.16   (kappa ~unconstrained, as theory demands)
mean corr(kappa, theta_E) : -0.96   (vs -0.93 at 20k, -0.83 at 8k low-SNR —
                                     posterior collapses onto the degeneracy
                                     ridge as data quality improves)
```

### Recovery with kappa (vs Run 6 without)
theta_E R² **0.94** vs 1.00 (degeneracy feeds uncertainty into theta_E);
kappa R² 0.21 (posterior ≈ prior); all others essentially unchanged
(e1 0.92, e2 0.90, gamma1 0.85, gamma2 0.80, x_s/y_s 0.99, R_s 0.98).

### Fiducial inference (kappa truth 0.05)
kappa posterior 0.082 ± 0.054 (spans most of the prior — honest); theta_E
1.108 ± 0.065 (8× wider than without kappa: the degeneracy at work).

**Figures:** full reference-style set in `figures/` (`*_kappa.png` variants).
**Report:** rewritten in the reference-report structure → `report/report.pdf`
(18 pages). Logs: scratchpad `logs/{gen8,train8,eval8,genk,traink,evalk}.log`.

---

## Open items / next steps

1. ~~Set SOURCE_AMP = 150~~ ✓ done (Run 4)
2. ~~Raise N_TRAIN to 20000~~ ✓ done (Run 4)
3. ~~Re-run kappa comparison at corrected SNR~~ ✓ done (Run 5)
4. ~~Raise N_TRAIN to 50000 + reference-style diagnostics + rewritten report~~
   ✓ done (Runs 6-7, report/report.pdf 18 pages)
5. Optional: the residual SBC underconfidence for theta_E/x_s/y_s/R_s did not
   respond to more data — if tighter calibration is wanted, raise SUMMARY_DIM
   (48 → 96) and/or COUPLING_DEPTH (4 → 6) and retrain.

---

## Where the raw sources live

- R² values: subplot titles in `figures/recovery.png`
- Old notebook with all saved CPU-run outputs: `git show main:notebooks/run_pipeline.ipynb`
- Report with full analysis: `report/report.pdf`
- Trained models: `data/models/*.keras` (git-ignored, reproducible)
