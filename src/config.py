"""
src/config.py
==============
NEW FILE — added while building out the full project.

WHAT: a single place that holds every "fixed number" the rest of the codebase
needs: the pixel grid, the PSF, the noise model, which physical values are
treated as known/fixed vs. inferred, the prior ranges p(theta), the dataset
sizes, and the neural-network/training hyper-parameters.

WHY: the simulator, the dataset generator, the training script and the
evaluation script all need to agree on things like "how many pixels is the
image" or "what order do the 9 lens parameters come in". Before this file
existed, those numbers were scattered (and partly hard-coded, e.g. inside
src/simulator/lens_generator.py) which makes it very easy for two parts of
the pipeline to silently disagree. Importing everything from here removes
that whole class of bug, and gives you ONE place to tune the project from.

HOW TO USE THIS FOR EXPERIMENTS
--------------------------------
Almost every "what if I change X" question in this project is answered by
editing a value in this file and re-running the relevant script. The comments
next to each block below say what changing that value does.
"""
import os

# ----------------------------------------------------------------------------
# 1. IMAGING SETUP (the simulated "telescope"). Fixed/known quantities.
# ----------------------------------------------------------------------------
NUM_PIX = 64          # image is NUM_PIX x NUM_PIX pixels.
DELTA_PIX = 0.05       # arcsec / pixel -> field of view = NUM_PIX * DELTA_PIX = 3.2"
PSF_FWHM = 0.10        # Gaussian PSF full-width-half-max, in arcsec (~2 px).
# TRY THIS: raising NUM_PIX (e.g. to 96) gives sharper images but makes every
# simulation and every CNN forward pass slower -- noticeable on a CPU-only
# machine. Lowering DELTA_PIX (finer pixels) has a similar cost/sharpness
# trade-off; raising it makes arcs blurrier and easier to learn but less
# realistic.

# ----------------------------------------------------------------------------
# 2. NOISE MODEL (fixed, known to us) -- Gaussian background + Poisson shot
#    noise, applied via lenstronomy's own noise utilities.
# ----------------------------------------------------------------------------
EXP_TIME = 100.0        # effective exposure time -> controls Poisson noise.
BACKGROUND_RMS = 0.10   # Gaussian sky/read noise standard deviation per pixel.
# TRY THIS: increasing BACKGROUND_RMS (e.g. to 0.3) or decreasing EXP_TIME
# makes the images noisier (lower SNR) -> the posteriors from BayesFlow
# should get *wider* (less certain) once you retrain. This is the easiest way
# to see "more noise = less information = broader posterior" for yourself.
# `python -m src.simulator.check_snr` (see below) reports the peak SNR so you
# can re-tune SOURCE_AMP / BACKGROUND_RMS back into a realistic 10-30 range.

# ----------------------------------------------------------------------------
# 3. FIXED (NUISANCE) PARAMETERS -- assumed already known, NOT inferred.
#    The assignment sheet says to fix: the lens centroid (known from the lens
#    light), the source brightness, and the source Sersic index/ellipticity.
# ----------------------------------------------------------------------------
LENS_CENTER_X = 0.0   # lens galaxy centroid, assumed known from the lens light.
LENS_CENTER_Y = 0.0
# TUNED (was 20.0, which gave median peak SNR ~4 -- well below the
# assignment's 10-30 target). 150.0 puts the median at ~16 (range ~6-21);
# see the amplitude-vs-SNR sweep in RESULTS.md.
SOURCE_AMP = 150.0    # source surface-brightness amplitude -- sets the SNR.
SOURCE_N = 2.0        # source Sersic index (shape of the light profile), fixed.
SOURCE_E1 = 0.05      # source-light ellipticity, fixed (only the LENS mass
SOURCE_E2 = -0.05     # ellipticity e1/e2 below is inferred, not this one).

# Rough scale used only to bring pixel values near O(1) before they enter the
# CNN. BatchNorm inside the network absorbs any remaining mis-scaling, so the
# exact number is not critical -- it just avoids feeding the network values
# in the thousands. Scaled up with SOURCE_AMP (peak pixel is now ~3).
IMAGE_SCALE = 4.0

# ----------------------------------------------------------------------------
# 4. WHAT WE INFER -> theta.
#    INCLUDE_KAPPA switches between the two versions the assignment asks for:
#      False -> 8 parameters, kappa fixed to 0 (the "simpler version").
#      True  -> 9 parameters, including the external convergence kappa.
#    Default is False so the local/CPU pipeline stays small; flip it to True
#    and re-run `generate` + `train` to reproduce the "with kappa" version and
#    compare (see notebooks/run_pipeline.ipynb, Section 7, for exactly this
#    comparison and the mass-sheet-degeneracy discussion it is meant to show).
# ----------------------------------------------------------------------------
# IMPORTANT: PARAM_NAMES/NUM_PARAMS just below are computed ONCE, right when
# this file is first imported. Editing INCLUDE_KAPPA here and re-running a
# script (`python main.py generate`) picks the change up correctly, because
# that starts a fresh Python process. But flipping `config.INCLUDE_KAPPA`
# from inside an already-running notebook cell will NOT update PARAM_NAMES
# retroactively (Python does not re-run this file just because you changed
# an attribute on the already-imported module) -- you would silently keep
# training/evaluating an 8-parameter model.
#
# The default (False) can be overridden with the SLI_INCLUDE_KAPPA=1
# environment variable, read once at import time, precisely so that
# notebooks/run_pipeline.ipynb can launch a *separate* `python main.py ...`
# subprocess with that variable set to get a clean, fully-independent 9
# parameter run -- without editing this file or restarting the notebook
# kernel. See notebooks/run_pipeline.ipynb, Section 7, for that comparison.
INCLUDE_KAPPA = os.environ.get("SLI_INCLUDE_KAPPA", "0") == "1"

# Order matters: this fixes the column order of every theta array/vector used
# anywhere in the project (dataset files, network outputs, plots, ...).
_BASE_PARAMS = ["theta_E", "e1", "e2", "gamma1", "gamma2", "x_s", "y_s", "R_s"]
PARAM_NAMES = _BASE_PARAMS + (["kappa"] if INCLUDE_KAPPA else [])
NUM_PARAMS = len(PARAM_NAMES)

# Human-readable math labels, used only for plot titles/axes.
PARAM_LABELS = {
    "theta_E": r"$\theta_E$", "e1": r"$e_1$", "e2": r"$e_2$",
    "gamma1": r"$\gamma_1$", "gamma2": r"$\gamma_2$",
    "x_s": r"$x_s$", "y_s": r"$y_s$", "R_s": r"$R_s$", "kappa": r"$\kappa$",
}

# ----------------------------------------------------------------------------
# 5. PRIOR RANGES p(theta) -- uniform unless noted. All lengths in arcsec,
#    all angles in radians. Train-time and test-time draws use these SAME
#    ranges, which is what makes the calibration check (SBC, in evaluation)
#    a valid diagnostic.
# ----------------------------------------------------------------------------
PRIOR = {
    "theta_E": (0.7, 1.6),        # Einstein radius: keeps the ring inside the FoV.
    "q": (0.6, 1.0),              # lens axis ratio -> converted to (e1, e2) below.
    "phi": (0.0, 3.14159265),     # lens position angle (rad) -> (e1, e2).
    "gamma_ext": (0.0, 0.08),     # external shear magnitude -> (gamma1, gamma2).
    "phi_ext": (0.0, 3.14159265), # external shear angle (rad).
    "x_s": (-0.3, 0.3),           # source x position.
    "y_s": (-0.3, 0.3),           # source y position.
    "R_s": (0.05, 0.25),          # source half-light radius (controls arc thickness).
    "kappa": (0.0, 0.20),         # external convergence -- only used if INCLUDE_KAPPA.
}
# TRY THIS: widening a range (e.g. theta_E to (0.4, 2.2)) makes the inference
# problem harder (more prior volume to distinguish) and usually needs more
# training data/epochs to reach the same accuracy. Narrowing a range makes
# the posterior recovery plots look "better" almost for free, because the
# task got easier -- worth remembering when judging your own results.

# ----------------------------------------------------------------------------
# 5b. FIDUCIAL SYSTEM -- one fixed "observed" lens used only for the final
#     inference figure (posterior vs. prior corner plot with truth markers),
#     mirroring the reference workflow's single-observation inference demo.
#     Values sit comfortably inside every prior range above and correspond to
#     a mildly elliptical lens (q ~ 0.85) with weak external shear and a
#     slightly off-center compact source.
# ----------------------------------------------------------------------------
FIDUCIAL_THETA = {
    "theta_E": 1.15, "e1": 0.06, "e2": -0.04,
    "gamma1": 0.02, "gamma2": 0.03,
    "x_s": 0.10, "y_s": -0.07, "R_s": 0.15,
    "kappa": 0.05,   # used only when INCLUDE_KAPPA is on.
}

# ----------------------------------------------------------------------------
# 6. DATASET SIZES.
#    These defaults are deliberately small: lenstronomy simulation is
#    CPU-bound (no GPU needed, but it is the slow step), and this project is
#    meant to run end-to-end on a laptop with no dedicated GPU. The
#    assignment's suggested budget is 1e4-1e5 simulations; if you have access
#    to a faster machine (or just want to let it run overnight) raise these.
# ----------------------------------------------------------------------------
# RAISED 8000 -> 20000 -> 50000 (the upper half of the assignment's 1e4-1e5
# budget). 20000 removed the overfitting seen in the 8000-sample runs; 50000
# further improved shear/ellipticity recovery. The mild SBC underconfidence
# of the best-constrained parameters persists at both 20k and 50k, i.e. it
# is a network-capacity effect, not a data-volume one (see RESULTS.md).
N_TRAIN = 50000
N_VAL = 2000
N_TEST = 300       # held-out set used only for the recovery/calibration plots.

# How many CPU processes to use while precomputing the dataset (the lens
# simulations are independent of each other, so this parallelizes almost
# perfectly). Sensible default: all logical cores minus one, so the machine
# stays responsive. Set to 1 to disable multiprocessing (useful for debugging).
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

DATA_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("data", "models")
# Filenames carry a "_kappa" suffix when INCLUDE_KAPPA is on, so a
# with-kappa run and a without-kappa run never overwrite each other's
# dataset/model files -- this is what lets notebooks/run_pipeline.ipynb
# Section 7 compare both versions side by side.
_suffix = "_kappa" if INCLUDE_KAPPA else ""
DATA_FILE = f"lens_dataset{_suffix}.npz"          # theta + images for train/val.
TEST_FILE = f"lens_testset{_suffix}.npz"          # held-out systems for diagnostics.
MODEL_FILE = os.path.join(MODEL_DIR, f"lens_approximator{_suffix}.keras")

# ----------------------------------------------------------------------------
# 7. NEURAL NETWORK / TRAINING HYPER-PARAMETERS.
#    Kept modest on purpose so a full training run finishes in well under an
#    hour on a CPU-only laptop with the default N_TRAIN above.
# ----------------------------------------------------------------------------
SUMMARY_DIM = 48       # size of the learned summary vector the CNN produces.
COUPLING_DEPTH = 4     # number of coupling layers in the normalizing flow.
EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
# TRY THIS: SUMMARY_DIM/COUPLING_DEPTH control network capacity. Bigger values
# can fit more complex posteriors but train slower and need more data to
# avoid overfitting -- with only a few thousand training images, bigger is
# not automatically better. EPOCHS is the cheapest thing to increase if
# training loss is still trending down when it stops (see the loss curve
# plotted at the end of training).

# Which Keras 3 backend to use. "torch" is already in requirements.txt and
# needs no extra setup: the Keras 3 torch backend automatically runs on the
# GPU whenever `torch.cuda.is_available()` is True (see requirements.txt for
# the CUDA-enabled wheel), and transparently falls back to CPU otherwise --
# nothing else in the project needs to change either way. src/models/train.py
# prints which device it picked at the start of training so you can confirm.
KERAS_BACKEND = "torch"

SEED = 42
