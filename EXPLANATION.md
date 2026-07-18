# Strong Lens Inference — The Complete Project Explanation

A plain-language walkthrough of the entire project: what problem it solves, how every
piece of the pipeline works, and — figure by figure — how to read each plot and what
ours actually shows. Numbers refer to the final **50,000-sample runs** (RESULTS.md,
Runs 6–7; `report/report.pdf`).

---

## 1. What is this project?

When a massive galaxy sits between us and a more distant galaxy, its gravity bends
the distant galaxy's light around it. Instead of one small blob, we see stretched
**arcs**, **multiple images**, or a complete **Einstein ring**. This is *strong
gravitational lensing*, and the distorted image is not random — it is a fingerprint
of the foreground galaxy's mass distribution.

**The task:** given a *single noisy telescope image* of such a system, recover the
physical parameters that produced it — with honest uncertainties, not just point
estimates.

**The approach:** *simulation-based inference* (SBI). We can easily simulate the
forward direction (parameters → image) with the `lenstronomy` package, but the
reverse direction has no usable likelihood formula. So we train a neural network
(with the `BayesFlow` framework) on tens of thousands of simulated
(parameters, image) pairs. After training once, the network outputs the full
Bayesian posterior distribution for *any* new image in milliseconds — this is
called **amortized** inference.

---

## 2. The physical model — what the 8 (or 9) parameters mean

Each simulated system is a **Singular Isothermal Ellipsoid (SIE)** lens galaxy with
**external shear**, magnifying an elliptical **Sérsic** source galaxy.

| Parameter | Symbol | What it controls in the image |
|---|---|---|
| Einstein radius | θ_E | The radius of the ring/arcs — directly tied to the lens mass |
| Lens ellipticity | e₁, e₂ | How squashed the lens mass is → distorts the arcs into a quadrupole pattern |
| External shear | γ₁, γ₂ | Tidal gravity from *neighbouring* structures → an additional subtle stretch |
| Source position | x_s, y_s | Where the background galaxy sits → decides arcs vs. ring vs. multiple images |
| Source size | R_s | The half-light radius of the source → arc thickness |
| External convergence | κ (optional 9th) | A uniform "sheet" of extra mass along the line of sight |

Everything else (lens centre, source brightness and profile shape, PSF, noise
levels) is fixed and assumed known.

**Why κ is special:** lensing theory proves that adding a uniform mass sheet while
rescaling the source leaves the image *almost unchanged* — the famous **mass-sheet
degeneracy**. κ therefore *cannot* be measured from one image, and a correct
inference method must *report* that impossibility (wide posterior) rather than
invent an answer. Testing whether our network does this is a central part of the
project.

---

## 3. The pipeline, step by step

```
priors ──► lenstronomy simulator ──► noise ──► offline dataset (.npz)
                                                    │
                                          BayesFlow training (GPU)
                                                    │
                              trained posterior network p̂(θ | image)
                                                    │
                 ┌──────────────┬──────────────┬────┴─────────┬──────────────┐
              recovery    calibration (SBC)  contraction   post. predictive  inference
```

1. **Priors** (`src/simulator/parameters.py`). Each parameter is drawn from a
   uniform range chosen so every draw is a physically valid, visibly lensed system
   (e.g. θ_E ∈ (0.7″, 1.6″) keeps the ring inside the image). Ellipticity and shear
   are drawn as (axis ratio, angle) and (modulus, angle) then converted to
   Cartesian components — this guarantees validity and creates the ring-shaped
   joint priors you see in the prior pairplot.
2. **Simulation** (`src/simulator/lens_generator.py`). lenstronomy ray-traces the
   source through the lens equation, convolves with a Gaussian PSF (FWHM 0.1″), and
   pixelates to 64×64 at 0.05″/pixel.
3. **Noise** (`src/simulator/noise.py`). Two components, like a real CCD: constant
   Gaussian background noise, plus Poisson shot noise that grows with the signal.
   The source brightness was tuned by a sweep so the peak signal-to-noise ratio is
   realistic (median ≈ 16, target range 10–30).
4. **Offline dataset** (`src/simulator/dataset.py`). 50,000 train / 2,000
   validation / 300 test systems, simulated in parallel on all CPU cores (~1
   minute) and stored, so training never touches the simulator.
5. **Training** (`src/models/`). Two networks learn jointly:
   - a **summary CNN** that compresses each 64×64 image into 48 learned numbers;
   - a **conditional coupling flow** (a normalizing flow) that turns those 48
     numbers into a full joint posterior distribution over θ.
   40 epochs, batch 32, Adam, early stopping armed (never triggered), ~2 h on an
   RTX 3060.
6. **Evaluation** (`src/evaluation/`). The diagnostics below, all computed on the
   300 held-out test systems with 2,000 posterior draws each.

Run it yourself: `python main.py generate` → `train` → `evaluate`
(set `SLI_INCLUDE_KAPPA=1` for the 9-parameter variant).

---

## 4. The plots, one by one — how to read each figure

All figures live in `figures/`; `*_kappa.png` versions are the same diagnostics for
the 9-parameter model.

### 4.1 `dataset_preview.png` — example training images
**What it is:** eight random noisy images from the training set with their true
Einstein radius.
**How to read it:** sanity check — the images should look like plausible lenses
(rings/arcs of different radii, positions, thicknesses), not blank noise.
**What ours shows:** clean variety of ring and arc morphologies; the arcs stand
clearly above the noise, consistent with the tuned SNR.

### 4.2 `prior_pairs.png` — the prior p(θ)
**What it is:** marginal histograms (diagonal) and pairwise joint distributions
(off-diagonal) of the parameter draws used to build the training set.
**How to read it:** documents exactly which universe of lenses the network was
trained on. Note the *ring-shaped* (e₁, e₂) and (γ₁, γ₂) joints — these come from
sampling (axis ratio, angle) / (modulus, angle) and converting to components; they
are correct by construction, not a bug.
**What ours shows:** flat, independent priors within the intended ranges, with the
expected ring-shaped Cartesian supports.

### 4.3 `training_loss.png` — the loss trajectory
**What it is:** training and validation loss (negative log-posterior density) per
epoch; thin lines are raw, thick lines are moving averages.
**How to read it:** the two curves should *decrease together*. Validation rising
while training keeps falling = overfitting (memorizing the training set).
Validation still falling at the last epoch = more epochs would help.
**What ours shows:** both curves fall monotonically for all 40 epochs, validation
finishing at −8.05 *below* the training loss — no overfitting whatsoever. (Our 8k
pilot *did* overfit; scaling the dataset fixed it. Early stopping was armed but
never needed.)

### 4.4 `recovery.png` — parameter recovery (the "does it work" plot)
**What it is:** one panel per parameter; x-axis = true value, y-axis = posterior
**median**, error bar = posterior **MAD** (median absolute deviation), dashed line
= perfect recovery, `r` = Pearson correlation. (`recovery_r2.png` is the same idea
with posterior mean ± std and R².)
**How to read it:** points hugging the diagonal = accurate point estimates. A flat
cloud = the image doesn't constrain that parameter (the posterior falls back to
the prior mean). Error-bar size shows how much uncertainty the posterior reports —
it should be large exactly where the scatter is large.
**What ours shows:**
| Parameter | r | R² | Reading |
|---|---|---|---|
| θ_E, x_s, y_s, R_s | 0.999 | 1.00 | essentially perfect — these dominate the image morphology |
| e₁ / e₂ | 0.970 / 0.960 | 0.94 / 0.92 | very good — the quadrupole imprint is measurable at this SNR |
| γ₁ / γ₂ | 0.928 / 0.895 | 0.86 / 0.80 | good — shear is intrinsically subtle (modulus ≤ 0.08); note the honest, wider error bars |

The hierarchy is physics, not failure: parameters that imprint strongly on the
image are recovered best.

### 4.5 `calibration_ecdf.png` — simulation-based calibration (the "honest error bars" plot)
**What it is:** for each test system, compute the fractional rank of the true value
among its 2,000 posterior samples. If the posterior is exactly right, these ranks
are uniform. The plot shows the *difference* between the empirical CDF of the ranks
and the uniform CDF, with a simultaneous 95% confidence band (grey ellipse).
**How to read it:**
- curve inside the band → calibrated (uncertainties trustworthy);
- **U/valley then peak going down-up-down (∪-shape in a histogram)** → ranks piled
  at the extremes → posterior too *narrow* → **overconfident** (dangerous);
- **S-shape (dip below, then rise above)** → ranks piled in the middle → posterior
  too *wide* → **underconfident** (conservative, safe).
**What ours shows:** e₁, e₂, γ₁, γ₂ stay inside the band — calibrated. θ_E, x_s,
y_s, R_s trace a mild S-shape that exits the band — their credible intervals are
slightly too wide. Two important nuances: (1) this is the *safe* direction of
error; (2) the shape is identical at 20k and 50k training samples, which proves it
is a **network-capacity** limitation, not a data limitation — the actionable fix is
a bigger summary vector or deeper flow, not more simulations.
(`calibration.png` shows the same test as classical rank histograms.)

### 4.6 `contraction.png` — posterior contraction vs. z-score (the "information gain" plot)
**What it is:** one point per test system per parameter.
- x-axis: **contraction** = 1 − posterior variance / prior variance. 0 = the image
  taught us nothing (posterior = prior); 1 = the image removed all prior
  uncertainty.
- y-axis: **posterior z-score** = (posterior mean − truth) / posterior std. Measures
  bias in units of the reported uncertainty.
**How to read it:** the ideal is the top of a funnel — contraction near 1, z-scores
spread tightly around 0 within ±2. Points at low contraction are fine *if* their
z-scores are still centred (uninformative but honest).
**What ours shows:** θ_E, x_s, y_s, R_s cluster at contraction ≈ 1 with |z| ≲ 2 —
maximal, unbiased information gain. Ellipticity and shear sit at intermediate
contraction — the image genuinely contains less information about them — with
z-scores still centred on zero: honest uncertainty, no bias.

### 4.7 `posterior_predictive.png` — the closed-loop check
**What it is:** take ONE observed test image (left panel), draw four independent
parameter samples from its posterior, push each back through the simulator
(noise-free), and display them side by side.
**How to read it:** if the posterior is concentrated on the right parameters, the
re-simulations should look like the observation. Differences *between* the four
draws visualize the residual posterior uncertainty.
**What ours shows:** ring radius, bright-knot positions, and azimuthal light
profile all reproduced; draws differ mainly in fine knot brightness — precisely
the detail that the pixel noise genuinely hides.

### 4.8 `inference_pairs.png` — inference on one "observed" system (the corner plot)
**What it is:** a full pairwise corner plot for a single fiducial lens
(θ_E = 1.15″, mildly elliptical, weak shear, off-centre compact source): grey =
prior, blue = 2,000 posterior draws, red = the true values used to simulate the
observation.
**How to read it:** the blue blob should (a) be much smaller than the grey cloud
(information gained), (b) contain the red cross (accuracy), and (c) reveal
correlations between parameters through its tilt.
**What ours shows:** dramatic contraction onto the truth for every parameter
(θ_E: 1.152 ± 0.008 vs. truth 1.150); the widest remaining spread is in the shear
components, consistent with every other diagnostic; visible tilted ellipses (e.g.
e₁–γ₁) show the network reports *joint* structure, not just per-parameter widths.

### 4.9 The κ variants and the mass-sheet degeneracy (`*_kappa.png`)
The same diagnostics for the 9-parameter model tell one coherent physics story:
- **`recovery_kappa.png`:** the κ panel is a flat cloud spanning the prior
  (R² = 0.21) — the network *cannot* measure κ, exactly as the theorem demands.
  θ_E degrades from R² 1.00 → 0.94 with visibly inflated error bars; every other
  parameter is untouched. The degeneracy is surgical.
- **Contraction of κ:** prior std 0.058 → mean posterior std 0.048, i.e. only
  +0.16 contraction. The network honestly reports "I don't know."
- **corr(κ, θ_E) = −0.96** within individual posteriors: the network discovered the
  exact trade-off direction the lens equations predict (more mass sheet ⇔ smaller
  intrinsic Einstein radius), and the correlation *strengthened* as data quality
  improved (−0.83 at low SNR → −0.96 now) because sharper posteriors hug the
  degeneracy ridge more tightly.
- **`inference_pairs_kappa.png`:** in the (κ, θ_E) panel you can *see* the tilted
  degeneracy ridge; the κ marginal barely contracts from its prior; θ_E's
  uncertainty is ~8× wider than in the 8-parameter fit of the same image
  (± 0.065 vs. ± 0.008) — the degeneracy feeding uncertainty into the mass.

Nobody told the network about the mass-sheet degeneracy. It *rediscovered* it from
data — the strongest evidence that the learned posteriors are physical.

### 4.10 `calibration_ecdf_20k.png`, `recovery_20k.png` — the comparison configuration
Preserved diagnostics of the intermediate 20,000-sample model, used in the report's
"Other Configurations" section to show which problems more data did and did not fix
(recovery ↑, calibration shape unchanged).

---

## 5. The headline results in one table

| Question | Diagnostic | Answer |
|---|---|---|
| Can it recover the parameters? | recovery | r = 0.999 for θ_E/x_s/y_s/R_s; 0.96–0.97 ellipticity; 0.90–0.93 shear |
| Are the error bars honest? | SBC rank-ECDF | Yes for e/γ; mildly conservative (never overconfident) for the rest |
| How much does one image teach us? | contraction/z-score | Near-total for morphology-dominant parameters, honest partial for shear |
| Does the posterior regenerate the data? | posterior predictive | Yes, down to knot positions |
| Does it respect known physics? | κ run | Mass-sheet degeneracy reproduced: κ unconstrained, corr(κ,θ_E) = −0.96 |
| Is it fast? | timing | Train once (~2 h GPU); then ~milliseconds per lens |

---

## 6. What we learned / limitations / next steps

- **SNR is destiny.** Fixing the source brightness from median peak SNR ≈ 4 to ≈ 16
  took shear recovery from unusable (R² 0.06) to good (0.80). No network can
  extract information the pixels don't contain.
- **Data volume fixes specific things.** 8k → 20k samples cured overfitting;
  20k → 50k sharpened recovery further. But the mild calibration conservatism did
  *not* respond to more data — a clean diagnosis that the remaining slack is
  **network capacity** (summary dim 48 / flow depth 4), which is the first knob to
  turn next.
- **Degeneracies are reported, not hidden.** The κ experiment shows the method
  fails *honestly* where physics says it must — the property that makes Bayesian
  outputs usable downstream.
- **Toward real data:** the simulator currently fixes the PSF, source profile, and
  lens light. Real survey images (Euclid, Rubin-LSST, ~10⁵ expected lenses) need
  those added to the simulator — the inference machinery stays identical, which is
  the whole appeal of SBI.

---

## 7. Where everything lives

| Artifact | Path |
|---|---|
| Full report (18 pp, reference structure) | `report/report.pdf` / `report.tex` |
| All figures | `figures/` |
| Slides + speech | `presentation/StrongLensInference.pptx`, `presentation/speech.md` |
| Every run's numbers | `RESULTS.md` |
| All tunable settings (one file) | `src/config.py` |
| Pipeline entry point | `main.py` (`generate` / `train` / `evaluate` / `demo`) |
