"""
src/models/train.py
=====================
NEW CONTENT (file previously existed but was empty). WHAT: trains the
BayesFlow posterior approximator (CNN summary net + coupling flow, from
bayesflow_model.py) on the dataset precomputed by
src/simulator/dataset.py / `python main.py generate`.

This is an OFFLINE training script: it never calls the lenstronomy simulator
directly -- it only reads the .npz arrays already saved to disk. That is the
whole point of precomputing the dataset separately (see dataset.py's
docstring): the slow, CPU-bound simulation step and the (comparatively
fast) neural-network training step are decoupled, so you can, for example,
generate data once and then experiment with several training configurations
without re-simulating anything.

Run (from the project root):
    python main.py train
or directly:
    python -m src.models.train
"""
import os

# Select the Keras 3 backend BEFORE keras or bayesflow is imported anywhere
# in the process -- Keras reads this environment variable exactly once, at
# import time. src/config.py's KERAS_BACKEND is the single place this is
# configured (default "torch", CPU-only, no GPU/CUDA needed).
from src import config as C
os.environ.setdefault("KERAS_BACKEND", C.KERAS_BACKEND)

import numpy as np
import keras
import bayesflow as bf

from src.models.bayesflow_model import build_summary_network, build_inference_network


def load_offline_data():
    """Load data/processed/lens_dataset.npz and reshape it into the dict
    layout BayesFlow expects: one (N, 1) array per parameter name, plus one
    (N, H, W, 1) array under the key "image"."""
    path = os.path.join(C.DATA_DIR, C.DATA_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `python main.py generate` first."
        )
    d = np.load(path, allow_pickle=True)

    def to_dict(theta, images):
        out = {name: theta[:, i:i + 1].astype(np.float32)
               for i, name in enumerate(C.PARAM_NAMES)}
        out["image"] = images.astype(np.float32)
        return out

    train = to_dict(d["theta_train"], d["images_train"])
    val = to_dict(d["theta_val"], d["images_val"])
    return train, val


def build_adapter():
    """Tell BayesFlow how to turn the raw dict above into network inputs:
      * stack the PARAM_NAMES columns into "inference_variables" (the
        targets the flow network learns to predict).
      * rename "image" to "summary_variables" (routed through the CNN
        before it ever reaches the flow network).
    """
    return (
        bf.Adapter()
        .convert_dtype("float64", "float32")
        .concatenate(C.PARAM_NAMES, into="inference_variables")
        .rename("image", "summary_variables")
    )


def build_workflow():
    """Assemble the BasicWorkflow: adapter + summary CNN + inference flow.

    `standardize=["inference_variables"]` z-scores the 8-9 physical
    parameters (which live on very different numeric scales, e.g. theta_E in
    arcsec vs. e1 in [-1, 1]) -- standardizing the *targets* like this is
    important for stable normalizing-flow training. The images are left
    as-is because the CNN already has its own Rescaling + BatchNorm layers
    to handle their scale (see bayesflow_model.py).
    """
    return bf.BasicWorkflow(
        adapter=build_adapter(),
        summary_network=build_summary_network(),
        inference_network=build_inference_network(),
        standardize=["inference_variables"],
    )


def print_device_info():
    """Report which device training will actually run on. The Keras 3 torch
    backend picks this automatically (GPU if `torch.cuda.is_available()`,
    else CPU) -- there is nothing else to configure, but it is easy to think
    a run is using the GPU when it silently isn't (e.g. CPU-only torch wheel
    installed), so print it explicitly every run."""
    print(f"Keras backend: {keras.backend.backend()}")
    if keras.backend.backend() == "torch":
        import torch
        if torch.cuda.is_available():
            print(f"Torch device: cuda ({torch.cuda.get_device_name(0)})")
        else:
            print("Torch device: cpu (no CUDA GPU detected -- see "
                  "requirements.txt if you expected GPU acceleration)")


def main():
    keras.utils.set_random_seed(C.SEED)

    print_device_info()
    print("Loading offline data ...")
    train_data, val_data = load_offline_data()
    print(f"  train images: {train_data['image'].shape}, "
          f"val images: {val_data['image'].shape}, "
          f"params: {C.PARAM_NAMES}")

    workflow = build_workflow()

    print(f"Training for {C.EPOCHS} epochs (batch_size={C.BATCH_SIZE}) ...")
    # TRY THIS: if training feels slow, the two cheapest local knobs are
    # config.EPOCHS (fewer epochs = faster but less accurate) and
    # config.N_TRAIN (fewer training images = both faster to simulate AND
    # faster per epoch). Both live in src/config.py.
    history = workflow.fit_offline(
        train_data,
        epochs=C.EPOCHS,
        batch_size=C.BATCH_SIZE,
        validation_data=val_data,
        verbose=2,
    )

    os.makedirs(C.MODEL_DIR, exist_ok=True)
    workflow.approximator.save(C.MODEL_FILE)
    print(f"Saved trained approximator -> {C.MODEL_FILE}")

    # Persist the loss curves. Previously only the notebook did this, so a
    # CLI run's losses existed nowhere but the terminal scrollback (see
    # RESULTS.md, Run 3). The "_kappa" suffix keeps the two model variants'
    # plots from overwriting each other.
    from src.evaluation import plots
    suffix = "_kappa" if C.INCLUDE_KAPPA else ""
    plots.plot_training_history(history, fname=f"training_loss{suffix}.png")

    # Returning both lets a notebook keep the trained workflow in memory and
    # go straight to evaluation (see notebooks/run_pipeline.ipynb) without a
    # separate save/load round trip, which is the most reliable way to run
    # evaluation right after training in the same session.
    return workflow, history


if __name__ == "__main__":
    main()
