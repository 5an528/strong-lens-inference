# Inferring Strong Gravitational Lens Parameters from Images

## Project Overview

This project aims to estimate the physical parameters of strong gravitational lens systems from simulated telescope images using **Simulation-Based Inference (SBI)** with **BayesFlow**.

Strong gravitational lensing occurs when the gravity of a massive foreground galaxy bends the light from a distant background galaxy, producing arcs, multiple images, or even complete Einstein rings. These observed patterns contain information about the mass distribution of the lens galaxy.

The project uses **Lenstronomy** to simulate realistic gravitational lens images and **BayesFlow** to infer the posterior distribution of the lens parameters from a single noisy image.

---

## Objectives

- Generate simulated strong gravitational lens images.
- Add realistic observational noise.
- Build a dataset of simulated images and corresponding physical parameters.
- Train a BayesFlow model to recover lens parameters.
- Evaluate the inference performance.
- Present the methodology and results.

---

## Technologies

- Python 3.11
- Lenstronomy
- BayesFlow
- PyTorch
- NumPy
- SciPy
- Matplotlib
- Astropy
- Scikit-learn

---

## Project Structure

```
StrongLensInference/
│
├── src/
│   ├── simulator/
│   │   ├── lens_generator.py
│   │   ├── parameters.py
│   │   └── noise.py
│   │
│   ├── models/
│   │   ├── bayesflow_model.py
│   │   └── train.py
│   │
│   └── evaluation/
│       ├── test.py
│       └── plots.py
│
├── data/
│   └── processed/          # generated datasets (git-ignored)
│
├── figures/
│
├── requirements.txt
├── README.md
└── main.py
```

---

## Installation

Clone the repository

```bash
git clone <repository_url>
cd StrongLensInference
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Workflow

1. Generate random lens parameters.
2. Simulate gravitational lens images using Lenstronomy.
3. Add Gaussian and Poisson noise.
4. Save image–parameter pairs.
5. Train the BayesFlow model.
6. Evaluate parameter estimation accuracy.
7. Visualize the results.

---

## Physical Parameters

The model estimates the following parameters:

| Parameter | Description |
|-----------|-------------|
| θE | Einstein Radius |
| e1 | Lens ellipticity component |
| e2 | Lens ellipticity component |
| γ1 | External shear |
| γ2 | External shear |
| xs | Source x-position |
| ys | Source y-position |
| Rs | Source radius |
| κ | External convergence (optional) |

For the simplified implementation, κ may be fixed to **0**.

---

## References

Lenstronomy Documentation

https://lenstronomy.readthedocs.io/en/latest/

BayesFlow Paper

https://arxiv.org/abs/1803.09746

Strong Gravitational Lensing Reference

https://arxiv.org/abs/1003.5567

---

## Timeline

- Environment setup
- Learn Lenstronomy
- Generate simulations
- Build dataset
- Train BayesFlow
- Evaluate results
- Prepare presentation

---

## Author

**Sayed Atique Newaz**

**Noureen Alam Meem**

**Mashuk Khan**


---

## License

This repository is created for academic purposes as part of a university project.

---

## Implementation Log -- what has been built (2026-07-10)

This section documents what exists in the codebase right now, on top of the original
project skeleton described above. Everything below is additive; nothing above this
line was rewritten. See the comment header at the top of every changed/new file for
the file-by-file "what changed and why".

### What's implemented

- **Full physical model.** The simulator now implements the model this project's
  assignment actually asks for: a Singular Isothermal Ellipsoid (SIE) lens + external
  SHEAR + an optional external CONVERGENCE sheet (kappa), lensing an elliptical Sersic
  source, PSF-convolved and pixelated with `lenstronomy`, corrupted with Gaussian
  background + Poisson shot noise. Parameters: `theta_E, e1, e2, gamma1, gamma2, x_s,
  y_s, R_s` (+ `kappa` when enabled) -- 8 or 9 numbers, matching the assignment's
  `theta`. This replaces the earlier single-parameter SIS + circular-source demo.
- **`src/config.py` (new).** One file holding every fixed number (grid, PSF, noise,
  which parameters are fixed vs. inferred, prior ranges, dataset sizes, network/training
  hyper-parameters) so the rest of the pipeline can never disagree about them.
- **Offline dataset generation** (`src/simulator/dataset.py`, new), parallelized across
  CPU cores (`config.N_WORKERS`) since simulation is the slow, CPU-bound step. Writes
  `data/processed/lens_dataset.npz` (train+val) and `lens_testset.npz` (held out).
- **BayesFlow model** (`src/models/bayesflow_model.py`, `src/models/train.py`): a small
  CNN summary network + a coupling-flow inference network, trained offline on the
  precomputed dataset. `main.py train` / `train.main()` returns the trained `workflow`
  object directly for immediate evaluation.
- **Evaluation suite** (`src/evaluation/plots.py`, `src/evaluation/test.py`): parameter
  recovery, simulation-based calibration (SBC), posterior predictive checks, and a
  kappa / mass-sheet-degeneracy analysis (the "with vs. without kappa" comparison the
  assignment asks about).
- **CLI entry point** (`main.py`, rewritten): `python main.py {demo, generate, train,
  evaluate}`. `demo` simulates and plots one random lens without needing a dataset --
  the fastest way to check the simulator still works after a config change.
- **Runs on CPU or GPU, no code changes needed.** The Keras 3 backend is `torch`;
  `requirements.txt` installs the CUDA-enabled wheel (cu124) by default, and the torch
  backend automatically trains on the GPU whenever `torch.cuda.is_available()` is True,
  falling back to CPU otherwise. `python main.py train` / `python -m src.models.train`
  print which device was picked at the start of each run. Dataset generation
  (lenstronomy) is always CPU-bound regardless -- only training and posterior sampling
  benefit from a GPU. If you don't have an NVIDIA GPU, swap the `--extra-index-url` line
  in `requirements.txt` for the CPU wheel index instead. A dedicated `.venv/` was set up
  inside this folder and all of the above was verified to run in it (dataset generation,
  training, evaluation, and the notebook).
- Generated datasets go to `data/processed/` (git-ignored, regenerate with
  `python main.py generate`).

### How to run it

```bash
cd strong-lens-inference
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python main.py demo                # fast sanity check, no dataset needed
python main.py generate            # precompute the dataset (a few minutes on CPU)
python main.py train                # train the BayesFlow approximator
python main.py evaluate             # recovery / calibration / posterior-predictive plots
```

### Where to change things

Every tunable number (noise level, prior ranges, `INCLUDE_KAPPA`, dataset size, network
size, training length, ...) lives in `src/config.py`, with a comment next to each value
explaining what changing it does.