"""
src/simulator/parameters.py
============================
REWRITTEN (was: a 14-line stub that only sampled theta_E, center_x, center_y
for a single-parameter SIS demo). WHY: the assignment asks for the full
9-parameter lens model theta = {theta_E, e1, e2, gamma1, gamma2, x_s, y_s,
R_s, kappa}, so the prior needs to draw all of those, not just three.

WHAT this file does: defines p(theta), the prior distribution the simulator
draws from to build the training/validation/test sets. Every simulated image
starts by calling `sample_prior()` here.

DESIGN NOTE: we do not draw the ellipticity components (e1, e2) or the shear
components (gamma1, gamma2) directly from uniform ranges. Instead we draw a
physically meaningful axis ratio q in (0, 1] and position angle phi (and,
for shear, a magnitude + angle), then convert to Cartesian components. This
guarantees every draw is a physically valid ellipse (e1^2+e2^2 < 1) while the
posterior BayesFlow learns is still expressed directly in (e1, e2, gamma1,
gamma2), exactly as the assignment specifies for theta.

This module only depends on numpy (no lenstronomy import), so it can be
tested/imported even in an environment where lenstronomy is not installed.
"""
import numpy as np

from src import config as C


def _ellipticity_from_q_phi(q, phi):
    """Axis ratio q + position angle phi (rad) -> (e1, e2).

    Uses lenstronomy's own convention (Util.param_util.phi_q2_ellipticity):
        e = (1-q) / (1+q);  e1 = e*cos(2*phi);  e2 = e*sin(2*phi)
    Reproduced here directly so this module has no lenstronomy dependency.
    """
    e = (1.0 - q) / (1.0 + q)
    return e * np.cos(2.0 * phi), e * np.sin(2.0 * phi)


def _shear_components(gamma_ext, phi_ext):
    """External shear magnitude + angle (rad) -> Cartesian (gamma1, gamma2)."""
    return gamma_ext * np.cos(2.0 * phi_ext), gamma_ext * np.sin(2.0 * phi_ext)


def sample_prior(rng=None):
    """Draw ONE parameter set theta ~ p(theta).

    Returns a dict mapping each name in config.PARAM_NAMES to a python float.
    This dict layout (one scalar per named key) is exactly what the
    lenstronomy wrapper in lens_generator.py and the BayesFlow simulator
    adapter expect.
    """
    rng = np.random.default_rng() if rng is None else rng
    P = C.PRIOR

    theta_E = rng.uniform(*P["theta_E"])

    q = rng.uniform(*P["q"])
    phi = rng.uniform(*P["phi"])
    e1, e2 = _ellipticity_from_q_phi(q, phi)

    gamma_ext = rng.uniform(*P["gamma_ext"])
    phi_ext = rng.uniform(*P["phi_ext"])
    gamma1, gamma2 = _shear_components(gamma_ext, phi_ext)

    x_s = rng.uniform(*P["x_s"])
    y_s = rng.uniform(*P["y_s"])
    R_s = rng.uniform(*P["R_s"])

    theta = {
        "theta_E": theta_E,
        "e1": e1, "e2": e2,
        "gamma1": gamma1, "gamma2": gamma2,
        "x_s": x_s, "y_s": y_s,
        "R_s": R_s,
    }
    if C.INCLUDE_KAPPA:
        theta["kappa"] = rng.uniform(*P["kappa"])

    return theta


def sample_prior_array(n, rng=None):
    """Draw n parameter sets as an (n, NUM_PARAMS) float array, with columns
    in config.PARAM_NAMES order. Convenient for building the offline dataset
    in one shot (see main.py's `generate` command)."""
    rng = np.random.default_rng() if rng is None else rng
    rows = [sample_prior(rng) for _ in range(n)]
    return np.array([[row[name] for name in C.PARAM_NAMES] for row in rows],
                     dtype=np.float64)


def theta_array_to_dict(theta_row):
    """Convert one (NUM_PARAMS,) row back into a name -> value dict. Used
    whenever we need to feed a theta *back* into the lenstronomy simulator,
    e.g. to re-simulate an image from a posterior sample."""
    return {name: float(theta_row[i]) for i, name in enumerate(C.PARAM_NAMES)}
