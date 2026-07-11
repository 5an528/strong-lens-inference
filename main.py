"""
main.py
========
REWRITTEN. WHY: this file used to hold a single hard-coded function,
`plot_simulated_lenses`, that only knew how to look at the old 3-parameter
SIS demo dataset in data/raw/. That plotting code has moved to
src/evaluation/plots.py (where the project's own folder layout says
plotting code belongs); this file is now the thin command-line entry point
the README's "Workflow" section describes, wiring the simulator, training
and evaluation modules together into runnable steps.

The OLD data/raw/lens_images.npy + lens_params.npy files are left on disk
untouched (first two git commits' output) but are NOT used by anything
below -- they came from the old 1-parameter SIS toy model and their shape
does not match the current 8/9-parameter SIE+SHEAR(+CONVERGENCE) model. New
datasets are written to data/processed/ instead (see src/config.py).

Usage (from the project root, with the virtual environment activated):
    python main.py generate    # precompute the simulated dataset (CPU, slow)
    python main.py train       # train the BayesFlow approximator (CPU, fast-ish)
    python main.py evaluate    # run recovery/calibration/posterior-predictive
                                # diagnostics on the held-out test set
    python main.py demo        # simulate + plot a single random lens, no
                                # dataset/training required -- the fastest way
                                # to sanity-check the simulator after any change.

For interactive exploration (recommended once you're past the "does this
run at all" stage) use notebooks/run_pipeline.ipynb instead, which runs the
same steps but keeps the trained `workflow` object in memory for evaluation.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def cmd_generate(args):
    from src.simulator.dataset import build_and_save
    build_and_save()


def cmd_train(args):
    from src.models.train import main as train_main
    train_main()


def cmd_evaluate(args):
    from src import config as C
    # Must set KERAS_BACKEND BEFORE the first `import keras` anywhere in the
    # process -- Keras picks its backend once, at import time, and otherwise
    # falls back to its own default (TensorFlow), which this project does
    # not install. See src/models/train.py for the same pattern.
    os.environ.setdefault("KERAS_BACKEND", C.KERAS_BACKEND)
    import keras
    from src.models.bayesflow_model import LensSummaryNet  # noqa: F401 (registers custom layer for loading)
    import bayesflow as bf
    from src.models.train import build_workflow, print_device_info
    from src.evaluation.test import run_all

    if not os.path.exists(C.MODEL_FILE):
        raise FileNotFoundError(
            f"{C.MODEL_FILE} not found. Run `python main.py train` first."
        )
    print_device_info()
    print(f"Loading trained approximator from {C.MODEL_FILE} ...")
    workflow = build_workflow()
    workflow.approximator = keras.saving.load_model(C.MODEL_FILE)
    run_all(workflow)


def cmd_demo(args):
    """Quick sanity check: simulate one random lens and plot it. Useful
    after editing src/config.py or the simulator to eyeball whether the
    images still look like sensible lenses before committing to a full
    (slow) dataset generation run."""
    from src.simulator.lens_generator import simulate_random_lens
    from src.simulator.noise import add_noise
    from src import config as C
    import matplotlib.pyplot as plt
    import numpy as np

    theta, clean = simulate_random_lens()
    noisy = add_noise(clean)

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(clean, origin="lower", cmap="magma")
    axes[0].set_title("clean (noise-free)")
    axes[0].axis("off")
    im = axes[1].imshow(noisy, origin="lower", cmap="magma")
    axes[1].set_title("with Gaussian + Poisson noise")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes.tolist(), label="pixel intensity", shrink=0.8)
    fig.savefig("figures/demo_single_lens.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print("theta =", {k: round(v, 3) for k, v in theta.items()})
    print("saved figures/demo_single_lens.png")


def main():
    parser = argparse.ArgumentParser(description="Strong gravitational lens SBI pipeline.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate", help="precompute the simulated dataset")
    sub.add_parser("train", help="train the BayesFlow approximator")
    sub.add_parser("evaluate", help="evaluate a trained approximator on the test set")
    sub.add_parser("demo", help="simulate and plot one random lens (no dataset needed)")

    args = parser.parse_args()
    {
        "generate": cmd_generate,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "demo": cmd_demo,
    }[args.command](args)


if __name__ == "__main__":
    main()
