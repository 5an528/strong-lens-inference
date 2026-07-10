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
import matplotlib.pyplot as plt

from src import config as C

FIGURES_DIR = "figures"


def plot_dataset_preview(images, theta, num_to_display=4, fname="dataset_preview.png"):
    """Show a grid of simulated lens images with their true theta_E in the
    title. Sanity-check plot: run this right after `generate` to make sure
    the images look like plausible lenses (rings/arcs, not noise or blanks)
    before spending time training.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    n = min(num_to_display, len(images))
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    tp = C.PARAM_NAMES.index("theta_E")
    for i in range(n):
        img = images[i, :, :, 0] if images.ndim == 4 else images[i]
        im = axes[i].imshow(img, origin="lower", cmap="magma")
        axes[i].set_title(f"Lens #{i + 1}\n" + r"$\theta_E$=" + f"{theta[i, tp]:.2f}\"")
        axes[i].axis("off")

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="pixel intensity")

    out = os.path.join(FIGURES_DIR, fname)
    plt.savefig(out, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_training_history(history, fname="training_loss.png"):
    """Plot the training/validation loss curves returned by
    workflow.fit_offline(). A validation loss that is still decreasing when
    training stops means more config.EPOCHS would likely still help; a
    validation loss that has flattened (or starts rising while train loss
    keeps falling) means the network is starting to overfit the training
    set -- more data (config.N_TRAIN) helps more than more epochs there.
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)
    hist = history.history if hasattr(history, "history") else history

    fig, ax = plt.subplots(figsize=(6, 4))
    if "loss" in hist:
        ax.plot(hist["loss"], label="train loss")
    if "val_loss" in hist:
        ax.plot(hist["val_loss"], label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (negative log-posterior density)")
    ax.legend()
    fig.tight_layout()

    out = os.path.join(FIGURES_DIR, fname)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"saved {out}")


def plot_recovery(theta_true, samples, fname="recovery.png"):
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
