"""
src/evaluation/plots.py
=========================
NEW CONTENT (file previously existed but was empty; the one plotting
function the project had, `plot_simulated_lenses`, was living in the
top-level main.py instead -- moved here, since main.py is meant to be a thin
CLI entry point and this is the "evaluation/plots" module by the project's
own folder layout). WHY MOVED: keeping all plotting code in one module means
notebooks and main.py both import from the same place instead of duplicating
plotting logic.

WHAT this file contains: every plotting function used to look at the
dataset and at the trained model's results. Nothing in here trains anything
or touches lenstronomy directly (except posterior_predictive, which
re-simulates from posterior samples to visually compare against the
observed image) -- it only reads arrays and produces figures under figures/.
"""
import os
import numpy as np
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingModuleSource]

from src import config as C

FIGURES_DIR = "figures"


def _labels():
    """Math-notation labels in PARAM_NAMES order, for BayesFlow's plots."""
    return [C.PARAM_LABELS[n] for n in C.PARAM_NAMES]


def _save(fig, fname, note=""):
    # BayesFlow's pair plots return a seaborn PairGrid rather than a bare
    # Figure -- unwrap to the underlying Figure so savefig/close work on both.
    if not isinstance(fig, plt.Figure):
        fig = getattr(fig, "figure", getattr(fig, "fig", fig))
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}" + (f"  ({note})" if note else ""))


def plot_dataset_preview(images, theta, num_to_display=8, fname="dataset_preview.png"):
    """Show a grid of simulated lens images with their true theta_E in the
    title (the analog of the reference report's "samples from the
    likelihood" figure). Sanity-check plot: run this right after `generate`
    to make sure the images look like plausible lenses (rings/arcs, not
    noise or blanks) before spending time training.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n = min(num_to_display, len(images))
    ncol = min(4, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.4 * nrow))
    axes = np.array(axes).ravel()

    tp = C.PARAM_NAMES.index("theta_E")
    for i in range(n):
        img = images[i, :, :, 0] if images.ndim == 4 else images[i]
        im = axes[i].imshow(img, origin="lower", cmap="magma")
        axes[i].set_title(r"$\theta_E$=" + f"{theta[i, tp]:.2f}\"", fontsize=11)
        axes[i].axis("off")
    for k in range(n, len(axes)):
        axes[k].axis("off")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="pixel intensity")

    out = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_training_history(history, fname="training_loss.png"):
    """Plot the training/validation loss curves returned by
    workflow.fit_offline(), in the "Loss Trajectory" style of BayesFlow's
    own loss plot: thin raw curves plus a thicker moving average. A
    validation loss that is still decreasing when training stops means more
    config.EPOCHS would likely still help; a validation loss that has
    flattened (or starts rising while train loss keeps falling) means the
    network is starting to overfit the training set -- more data
    (config.N_TRAIN) helps more than more epochs there.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    hist = history.history if hasattr(history, "history") else history

    def _ma(x, w=5):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return x
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = np.arange(1, len(hist["loss"]) + 1)
    if "loss" in hist:
        ax.plot(epochs, hist["loss"], color="#132a70", alpha=0.35, lw=1,
                label="Training")
        ma = _ma(hist["loss"])
        ax.plot(epochs[len(epochs) - len(ma):], ma, color="#132a70", lw=2,
                label="Training (Moving Average)")
    if "val_loss" in hist:
        ax.plot(epochs, hist["val_loss"], ".--", color="grey", alpha=0.6,
                lw=1, ms=4, label="Validation")
        ma = _ma(hist["val_loss"])
        ax.plot(epochs[len(epochs) - len(ma):], ma, "--", color="black", lw=1.8,
                label="Validation (Moving Average)")
    ax.set_title("Loss Trajectory")
    ax.set_xlabel("Training epoch #")
    ax.set_ylabel("Loss (negative log-posterior density)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_recovery(theta_true, samples, fname="recovery.png"):
    """Reference-style parameter recovery via BayesFlow's own diagnostic:
    posterior MEDIAN vs. ground truth with the median absolute deviation
    (MAD) as the uncertainty bar and the Pearson correlation r annotated in
    each panel -- exactly the recovery figure style of the BayesFlow
    workflow papers. Points on the dashed diagonal = perfect recovery.
    """
    import bayesflow as bf

    fig = bf.diagnostics.plots.recovery(
        estimates=samples, targets=theta_true, variable_names=_labels(),
    )
    _save(fig, fname, "posterior median vs truth, MAD error bars, Pearson r")


def plot_recovery_r2(theta_true, samples, fname="recovery_r2.png"):
    """Posterior-mean-vs-true-value scatter, one panel per parameter, with
    the posterior standard deviation as an error bar. Points on the black
    dashed diagonal = perfect recovery. This is the main "did it learn
    anything" plot: a high R^2 and small scatter around the diagonal means
    the network recovers that parameter well from the image; a flat cloud
    (R^2 near 0) means the image barely constrains it (see kappa_analysis
    in test.py for the textbook example of this: external convergence).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    means = samples.mean(axis=1)      # (N, P)
    stds = samples.std(axis=1)        # (N, P)

    n = C.NUM_PARAMS
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.2 * nrow))
    axes = np.array(axes).ravel()
    for p, name in enumerate(C.PARAM_NAMES):
        ax = axes[p]
        ax.errorbar(theta_true[:, p], means[:, p], yerr=stds[:, p],
                    fmt=".", alpha=0.35, ms=4, elinewidth=0.6)
        lo = min(theta_true[:, p].min(), means[:, p].min())
        hi = max(theta_true[:, p].max(), means[:, p].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        r2 = _r2(theta_true[:, p], means[:, p])
        ax.set_title(f"{C.PARAM_LABELS[name]}   $R^2$={r2:.2f}")
        ax.set_xlabel("true")
        ax.set_ylabel("posterior mean")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def _r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def plot_calibration_ecdf(theta_true, samples, fname="calibration_ecdf.png"):
    """Simulation-based calibration as a rank-ECDF *difference* plot with
    simultaneous 95% confidence bands (Saeilynoja et al. 2022), via
    BayesFlow's own diagnostic -- the same SBC figure style as the reference
    workflow. The difference between the empirical CDF of the fractional
    ranks and the uniform CDF should stay inside the grey band; excursions
    below/above indicate over/underconfident or biased posteriors.
    """
    import bayesflow as bf

    fig = bf.diagnostics.plots.calibration_ecdf(
        estimates=samples, targets=theta_true, variable_names=_labels(),
        difference=True,
    )
    _save(fig, fname, "inside the band = calibrated")


def plot_contraction(theta_true, samples, fname="contraction.png"):
    """Posterior z-score vs. posterior contraction, via BayesFlow's own
    diagnostic. Contraction = 1 - posterior variance / prior variance
    (near 1 = strong information gain from the image); the z-score is the
    standardized offset of the posterior mean from the truth (should spread
    around 0, mostly within +-2). The ideal regime is the top-right funnel:
    high contraction, small |z|.
    """
    import bayesflow as bf

    fig = bf.diagnostics.plots.z_score_contraction(
        estimates=samples, targets=theta_true, variable_names=_labels(),
    )
    _save(fig, fname, "contraction near 1 + z near 0 = informative, unbiased")


def plot_prior_pairs(theta, max_points=1000, fname="prior_pairs.png"):
    """Marginal + pairwise joint distributions of the prior draws used to
    build the training set (the analog of the reference report's prior
    pairplot). Documents p(theta) visually, including the ring-shaped
    (e1, e2) and (gamma1, gamma2) supports induced by sampling (q, phi) and
    (gamma_ext, phi_ext) and converting to Cartesian components.
    """
    import bayesflow as bf

    fig = bf.diagnostics.plots.pairs_samples(
        samples=theta[:max_points], variable_names=_labels(), label="Prior",
    )
    _save(fig, fname)


def plot_inference_pairs(post_samples, prior_samples, targets,
                          fname="inference_pairs.png"):
    """The 'inference on one observed system' figure: pairwise posterior
    (blue) vs. prior (grey) with the true parameter values marked (red),
    via BayesFlow's pairs_posterior -- the analog of the reference report's
    posterior-vs-prior corner plot for a single fiducial observation.

    post_samples: (S, P) posterior draws for the one observed image.
    prior_samples: (M, P) draws from p(theta).
    targets: (P,) true parameter values used to simulate the observation.
    """
    import bayesflow as bf

    fig = bf.diagnostics.plots.pairs_posterior(
        estimates=post_samples[None, ...],
        targets=np.asarray(targets)[None, :],
        priors=prior_samples,
        dataset_id=0,
        variable_names=_labels(),
    )
    _save(fig, fname, "posterior vs prior for the fiducial system")


def plot_calibration(theta_true, samples, fname="calibration.png"):
    """Simulation-based calibration (SBC, Talts et al. 2018): for each
    held-out system, compute the rank of the true theta among its own
    posterior samples. If the posterior is well-calibrated, these ranks are
    uniformly distributed -- i.e. the true value is, on average, equally
    likely to land anywhere within the posterior spread, never
    systematically too high or too low. The histogram should look flat and
    stay within the grey band; a U-shape means the posterior is
    overconfident (too narrow), a dome/hump shape means it is
    underconfident (too wide) or biased.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    N, S, P = samples.shape
    ranks = (samples < theta_true[:, None, :]).sum(axis=1)  # (N, P) in [0, S]

    n = C.NUM_PARAMS
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.array(axes).ravel()
    nbins = 20
    expected = N / nbins
    band = 2 * np.sqrt(expected * (1 - 1 / nbins))  # ~95% band for a flat histogram
    for p, name in enumerate(C.PARAM_NAMES):
        ax = axes[p]
        ax.hist(ranks[:, p] / S, bins=nbins, range=(0, 1),
                color="steelblue", alpha=0.8)
        ax.axhspan(expected - band, expected + band, color="grey", alpha=0.25)
        ax.axhline(expected, color="k", lw=1)
        ax.set_title(C.PARAM_LABELS[name])
        ax.set_xlabel("rank (normalized)")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out} (flat within the grey band = well calibrated)")


def plot_posterior_predictive(image, samples_theta, k=4, fname="posterior_predictive.png"):
    """Re-simulate k posterior draws for ONE observed image and show them
    side by side with the observation. If the model learned the physics
    well, the re-simulated images should look visually similar to the
    observed one (same ring/arc pattern), even though the network never saw
    this exact image during training.

    `samples_theta` is the (S, P) posterior-sample array for that one image
    (already sampled by src.evaluation.test.sample_posteriors).
    """
    from src.simulator.lens_generator import simulate_clean
    from src.simulator.parameters import theta_array_to_dict

    os.makedirs(FIGURES_DIR, exist_ok=True)
    idx = np.random.choice(samples_theta.shape[0], size=k, replace=False)

    fig, axes = plt.subplots(1, k + 1, figsize=(3 * (k + 1), 3))
    obs = image[:, :, 0] if image.ndim == 3 else image
    axes[0].imshow(obs, origin="lower", cmap="magma")
    axes[0].set_title("observed")
    axes[0].axis("off")
    for j, i in enumerate(idx):
        theta_dict = theta_array_to_dict(samples_theta[i])
        sim = simulate_clean(theta_dict)
        axes[j + 1].imshow(sim, origin="lower", cmap="magma")
        axes[j + 1].set_title(f"posterior draw {j + 1}")
        axes[j + 1].axis("off")
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")
