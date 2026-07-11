"""
src/simulator/lens_generator.py
=================================
REWRITTEN (was: a single-parameter demo using an SIS lens + a circular
SERSIC source, with no PSF, no shear, no convergence). WHY: the assignment
asks for a Singular Isothermal Ellipsoid (SIE) main deflector plus external
SHEAR and a uniform external CONVERGENCE, lensing an elliptical Sersic
source, convolved with a fixed Gaussian PSF -- none of which the stub did.

WHAT this file does: the forward model theta -> noisy image,
    theta = {theta_E, e1, e2, gamma1, gamma2, x_s, y_s, R_s, [kappa]}
built with lenstronomy. This is the "simulator" in simulation-based
inference: BayesFlow never sees a likelihood, only (theta, image) pairs
produced by calling this module many times over prior draws.

Physics recap
--------------
A background source galaxy (an elliptical Sersic blob) is lensed by a
foreground galaxy. The foreground mass is modelled as:
    * SIE         -- Singular Isothermal Ellipsoid, the main deflector
                     (theta_E, e1, e2, plus a FIXED known centroid).
    * SHEAR       -- external shear from line-of-sight structure
                     (gamma1, gamma2).
    * CONVERGENCE -- a uniform "mass sheet" of external convergence (kappa),
                     only added when config.INCLUDE_KAPPA is True.
Light from the source is ray-traced through this mass distribution,
convolved with a Gaussian PSF, sampled onto a fixed pixel grid, and only
*after* that is noise added (see noise.py) -- exactly the assignment's
"Build the ImageModel once and reuse it across draws, varying only the
parameter dictionaries" recipe.

Everything FIXED (grid, PSF, source brightness/index/ellipticity, lens
centroid) comes from src/config.py. Everything in `theta` varies per draw.
The heavy lenstronomy objects (pixel grid, PSF, LensModel, LightModel) are
built once and cached in `_STATIC`, so simulating 10^4-10^5 images does not
pay their setup cost every time -- this is the main reason dataset
generation is fast enough to run on a CPU-only laptop.
"""
import numpy as np

from src import config as C
from src.simulator.parameters import sample_prior

# lenstronomy is imported lazily inside _get_static() so this module can
# still be imported (e.g. to read PARAM_NAMES-adjacent helpers) in contexts
# where lenstronomy is not installed yet.
_STATIC = None


def _get_static():
    """Build & cache the pixel grid, PSF, LensModel and LightModel. Called
    once; every later simulate_clean() call reuses these objects and only
    swaps in new theta-dependent keyword-argument dictionaries."""
    global _STATIC
    if _STATIC is not None:
        return _STATIC

    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LightModel.light_model import LightModel
    from lenstronomy.Data.imaging_data import ImageData
    from lenstronomy.Data.psf import PSF
    from lenstronomy.ImSim.image_model import ImageModel
    import lenstronomy.Util.simulation_util as sim_util

    # (a) Pixel grid: a NUM_PIX x NUM_PIX detector at DELTA_PIX arcsec/pixel.
    kwargs_data = sim_util.data_configure_simple(C.NUM_PIX, C.DELTA_PIX)
    data_class = ImageData(**kwargs_data)

    # (b) Fixed Gaussian PSF.
    kwargs_psf = {"psf_type": "GAUSSIAN", "fwhm": C.PSF_FWHM,
                  "pixel_size": C.DELTA_PIX}
    psf_class = PSF(**kwargs_psf)

    # (c) Lens mass model: SIE + external SHEAR (+ optional CONVERGENCE sheet).
    lens_list = ["SIE", "SHEAR"] + (["CONVERGENCE"] if C.INCLUDE_KAPPA else [])
    lens_model_class = LensModel(lens_model_list=lens_list)

    # (d) Source light model: a single elliptical Sersic.
    source_model_class = LightModel(light_model_list=["SERSIC_ELLIPSE"])

    # (e) Tie it together. supersampling=1 keeps simulation fast on CPU;
    # raise supersampling_factor (e.g. to 3) for higher fidelity at the cost
    # of speed if you want more realistic images and can afford slower runs.
    kwargs_numerics = {"supersampling_factor": 1,
                        "supersampling_convolution": False}
    image_model = ImageModel(data_class, psf_class,
                              lens_model_class=lens_model_class,
                              source_model_class=source_model_class,
                              kwargs_numerics=kwargs_numerics)

    _STATIC = {"image_model": image_model}
    return _STATIC


def _kwargs_from_theta(theta):
    """Translate a theta dict (see parameters.sample_prior) into the
    keyword-argument lists lenstronomy's ImageModel.image() expects."""
    kwargs_sie = {
        "theta_E": theta["theta_E"], "e1": theta["e1"], "e2": theta["e2"],
        "center_x": C.LENS_CENTER_X, "center_y": C.LENS_CENTER_Y,
    }
    kwargs_shear = {"gamma1": theta["gamma1"], "gamma2": theta["gamma2"]}
    kwargs_lens = [kwargs_sie, kwargs_shear]
    if C.INCLUDE_KAPPA:
        # NOTE: some lenstronomy versions name the CONVERGENCE parameter
        # "kappa" and others "kappa_ext". If this raises a parameter error
        # after a lenstronomy upgrade, change the key below accordingly.
        kwargs_lens.append({"kappa": theta["kappa"]})

    kwargs_source = [{
        "amp": C.SOURCE_AMP, "R_sersic": theta["R_s"], "n_sersic": C.SOURCE_N,
        "e1": C.SOURCE_E1, "e2": C.SOURCE_E2,
        "center_x": theta["x_s"], "center_y": theta["y_s"],
    }]
    return kwargs_lens, kwargs_source


def simulate_clean(theta):
    """Noise-free (but PSF-convolved, pixelated) image for a theta dict.
    Shape (config.NUM_PIX, config.NUM_PIX)."""
    image_model = _get_static()["image_model"]
    kwargs_lens, kwargs_source = _kwargs_from_theta(theta)
    return image_model.image(kwargs_lens, kwargs_source)


def simulate_random_lens(rng=None):
    """Convenience helper for quick demos/notebooks: draw theta ~ p(theta)
    and return (theta_dict, clean_image) without adding noise."""
    theta = sample_prior(rng)
    return theta, simulate_clean(theta)
