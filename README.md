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
│   ├── raw/
│   └── processed/
│
├── notebooks/
├── figures/
├── presentation/
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