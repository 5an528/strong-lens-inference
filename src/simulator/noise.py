"""
src/simulator/noise.py
========================
REWRITTEN (was: 6 lines adding only plain Gaussian noise with a fixed
scale=1.5, unrelated to any physical exposure time). WHY: the assignment
asks for "a Gaussian background plus Poisson noise via lenstronomy's
utilities, using a fixed exposure time and background RMS" -- real telescope
images have both a constant sky/read-noise floor (Gaussian) AND shot noise
from the finite number of photons collected, which scales with the signal
itself (Poisson/shot noise, brighter pixels are noisier in absolute terms).

WHAT this file does: adds that two-component noise model on top of a clean
lenstronomy image, and provides a `peak_snr` helper used by
generate_dataset.py to report whether the images are in a realistic
signal-to-noise regime.
"""
import numpy as np

from src import config as C


def add_noise(clean_image, rng=None):
    """Add Gaussian background + Poisson shot noise, using config's
    EXP_TIME / BACKGROUND_RMS. Matches lenstronomy's own noise convention
    (image_util.add_background / add_poisson) so simulated noise behaves
    like real CCD/telescope noise rather than a single ad-hoc sigma.
    """
    from lenstronomy.Util import image_util

    rng = np.random.default_rng() if rng is None else rng
    # lenstronomy's noise helpers draw from numpy's *global* RNG state, so we
    # seed that global state from our own rng to keep runs reproducible
    # without giving up lenstronomy's tested noise implementation.
    np.random.seed(rng.integers(0, 2**31 - 1))

    background_noise = image_util.add_background(clean_image, sigma_bkd=C.BACKGROUND_RMS)
    poisson_noise = image_util.add_poisson(clean_image, exp_time=C.EXP_TIME)
    return clean_image + background_noise + poisson_noise


def peak_snr(clean_image):
    """Rough peak signal-to-noise ratio of a clean (noise-free) image.

    noise variance per pixel ~= background^2 + signal / exp_time (Poisson).
    Used only for diagnostics: generate_dataset.py prints the median peak
    SNR across the dataset so you can tune config.SOURCE_AMP /
    config.BACKGROUND_RMS until it sits in the realistic ~10-30 range quoted
    in the assignment.
    """
    peak = float(np.max(clean_image))
    noise = np.sqrt(C.BACKGROUND_RMS ** 2 + max(peak, 0.0) / C.EXP_TIME)
    return peak / noise if noise > 0 else 0.0
