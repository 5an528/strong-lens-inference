"""
src/evaluation/test.py
========================
NEW CONTENT (file previously existed but was empty). NOTE ON THE NAME: this
file is called "test.py" because that is what the original project skeleton
named it (see README's project-structure listing) -- despite the name, it is
NOT a pytest unit-test suite, it is the model-criticism / evaluation script:
"does the trained posterior actually work on held-out data?". Plain
"test.py" (no test_ prefix) is not auto-collected by pytest, so this is safe
to keep, but be aware of the naming if you ever add real unit tests too
(name those e.g. test_priors.py so they're both collected correctly and not
confused with this file).

WHAT this file does, matching the assignment's evaluation requirements:
  1. sample_posteriors  : draw posterior samples for a batch of images.
  2. kappa_analysis      : the "mass-sheet degeneracy" check the assignment
                            specifically asks about when INCLUDE_KAPPA=True.
  3. run_all              : the one function you actually call -- runs every
                            diagnostic in src/evaluation/plots.py against the
                            held-out test set and saves the figures.

Run (right after training, in the SAME process/session, e.g. from
notebooks/run_pipeline.ipynb) so the trained `workflow` object stays in
memory -- reloading a saved model from disk works too but the notebook path
is the most reliable, matching how BayesFlow workflows are typically used.
"""
import os
import numpy as np

from src import config as C
from src.evaluation import plots


def _stack_samples(post):
    """Turn BayesFlow's workflow.sample(...) output (a dict of per-parameter
    arrays) into a single (N, S, P) array with columns in PARAM_NAMES order,
    which is the shape every plotting function in plots.py expects."""
    if isinstance(post, dict) and all(n in post for n in C.PARAM_NAMES):
        cols = []
        for name in C.PARAM_NAMES:
            arr = np.asarray(post[name])
            if arr.ndim == 2:  # (N, S) -> (N, S, 1)
                arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
            cols.append(arr)
        return np.concatenate(cols, axis=-1)
    # Fallback: some BayesFlow versions can return one concatenated array.
    return np.asarray(post["inference_variables"] if isinstance(post, dict) else post)


def sample_posteriors(workflow, images, num_samples=2000):
    """Draw `num_samples` posterior samples for each image in `images`.
    Returns an (N_images, num_samples, NUM_PARAMS) array.

    TRY THIS: num_samples trades plot smoothness for speed -- 2000 gives
    smooth calibration histograms; drop it to a few hundred for a quick
    look while iterating.
    """
    post = workflow.sample(conditions={"image": images.astype(np.float32)},
                            num_samples=num_samples)
    return _stack_samples(post)


def kappa_analysis(theta_true, samples):
    """Quantify the external-convergence / mass-sheet degeneracy.

    Physical background: a uniform convergence sheet (kappa) mostly rescales
    the *source* rather than changing the observed image morphology, so a
    single lensed image barely constrains kappa on its own -- this is the
    classic "mass-sheet degeneracy". We expect:
      * the posterior for kappa to stay almost as wide as the prior
        (little "contraction"), and
      * a strong correlation between kappa and theta_E within each
        posterior (the two trade off against each other).
    Only meaningful when config.INCLUDE_KAPPA=True; this is exactly the
    "with vs. without kappa" comparison the assignment asks you to look at.
    """
    if "kappa" not in C.PARAM_NAMES:
        print("  kappa not in the model (INCLUDE_KAPPA=False in config.py) "
              "-- nothing to analyze. Flip INCLUDE_KAPPA to True, re-run "
              "`generate` + `train`, and call this again to see the "
              "mass-sheet degeneracy.")
        return
    kp = C.PARAM_NAMES.index("kappa")
    tp = C.PARAM_NAMES.index("theta_E")

    prior_lo, prior_hi = C.PRIOR["kappa"]
    prior_std = (prior_hi - prior_lo) / np.sqrt(12)          # std of a uniform prior
    post_std = samples[:, :, kp].std(axis=1).mean()          # mean posterior std

    corrs = []
    for i in range(samples.shape[0]):
        c = np.corrcoef(samples[i, :, kp], samples[i, :, tp])[0, 1]
        if np.isfinite(c):
            corrs.append(c)

    print("  --- kappa (external convergence) / mass-sheet degeneracy ---")
    print(f"    prior std          : {prior_std:.3f}")
    print(f"    mean posterior std : {post_std:.3f}   "
          f"(contraction = {1 - post_std / prior_std:+.2f}; near 0 = "
          f"posterior barely narrower than the prior, i.e. kappa is poorly "
          f"constrained by a single image, as expected)")
    print(f"    mean corr(kappa, theta_E) within posterior : {np.mean(corrs):+.2f}"
          f"  (a strongly negative/positive value is the signature of the "
          f"degeneracy trading kappa off against the mass normalization)")


def fiducial_inference(workflow, num_samples=2000, num_prior=1000):
    """Simulate ONE 'observed' lens from the fixed fiducial parameters in
    config.FIDUCIAL_THETA, draw posterior samples for it, and save the
    posterior-vs-prior corner plot with the truth marked (the reference
    workflow's single-observation inference figure). Returns the posterior
    samples so callers can print summary statistics.
    """
    from src.simulator.lens_generator import simulate_clean
    from src.simulator.noise import add_noise
    from src.simulator.parameters import sample_prior_array

    suffix = "_kappa" if C.INCLUDE_KAPPA else ""
    theta_fid = np.array([C.FIDUCIAL_THETA[n] for n in C.PARAM_NAMES],
                          dtype=np.float32)
    rng = np.random.default_rng(C.SEED + 7_000_000)
    clean = simulate_clean({n: float(v) for n, v in zip(C.PARAM_NAMES, theta_fid)})
    obs = add_noise(clean, rng=rng).astype(np.float32)[None, :, :, None]

    print("Fiducial-system inference (posterior vs prior corner plot) ...")
    post = sample_posteriors(workflow, obs, num_samples=num_samples)[0]  # (S, P)
    prior = sample_prior_array(num_prior, rng=rng).astype(np.float32)

    plots.plot_inference_pairs(post, prior, theta_fid,
                                fname=f"inference_pairs{suffix}.png")
    print("  fiducial truth vs posterior mean +- std:")
    for i, name in enumerate(C.PARAM_NAMES):
        print(f"    {name:>8}: truth={theta_fid[i]:+.3f}   "
              f"post={post[:, i].mean():+.3f} +- {post[:, i].std():.3f}")
    return post


def run_all(workflow, num_samples=2000):
    """Run every evaluation diagnostic against the held-out test set and
    save all figures under figures/. This is the single call you need after
    training: `from src.evaluation.test import run_all; run_all(workflow)`.
    """
    test_path = os.path.join(C.DATA_DIR, C.TEST_FILE)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"{test_path} not found. Run `python main.py generate` first.")
    d = np.load(test_path, allow_pickle=True)
    theta_true = d["theta_test"].astype(np.float32)
    images = d["images_test"].astype(np.float32)

    suffix = "_kappa" if C.INCLUDE_KAPPA else ""

    print("Sampling posteriors for the held-out test set ...")
    samples = sample_posteriors(workflow, images, num_samples=num_samples)

    print("Diagnostics:")
    plots.plot_recovery(theta_true, samples, fname=f"recovery{suffix}.png")
    plots.plot_recovery_r2(theta_true, samples, fname=f"recovery_r2{suffix}.png")
    plots.plot_calibration_ecdf(theta_true, samples,
                                 fname=f"calibration_ecdf{suffix}.png")
    plots.plot_calibration(theta_true, samples, fname=f"calibration{suffix}.png")
    plots.plot_contraction(theta_true, samples, fname=f"contraction{suffix}.png")
    plots.plot_posterior_predictive(images[0], samples[0],
                                     fname=f"posterior_predictive{suffix}.png")
    kappa_analysis(theta_true, samples)

    # Dataset-documentation figures (prior pairplot + example images), built
    # from the training file so the report can show what the network saw.
    train_path = os.path.join(C.DATA_DIR, C.DATA_FILE)
    if os.path.exists(train_path):
        dt = np.load(train_path, allow_pickle=True)
        plots.plot_prior_pairs(dt["theta_train"],
                                fname=f"prior_pairs{suffix}.png")
        plots.plot_dataset_preview(dt["images_train"][:8], dt["theta_train"][:8],
                                    num_to_display=8,
                                    fname=f"dataset_preview{suffix}.png")

    fiducial_inference(workflow, num_samples=num_samples)
