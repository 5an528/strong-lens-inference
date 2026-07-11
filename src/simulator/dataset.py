"""
src/simulator/dataset.py
==========================
NEW FILE. WHY: the assignment explicitly recommends precomputing 1e4-1e5
simulations OFFLINE (since lenstronomy runs on CPU and is the slow part of
the whole pipeline) and then training on the stored arrays. This file is
that offline precompute step: it draws theta ~ p(theta) many times, runs the
forward model + noise, and saves everything to compressed .npz files under
data/processed/ so src/models/train.py never has to touch lenstronomy again.

WHY MULTIPROCESSING: on a CPU-only machine, simulating tens of thousands of
64x64 lens images one at a time is the single slowest part of this project.
Each simulation is independent of the others (different random theta), so we
parallelize across CPU cores with `ProcessPoolExecutor`. On a 12-core laptop
this turns e.g. a 20-minute single-core run into roughly 2-3 minutes. Set
config.N_WORKERS = 1 to fall back to a simple single-process loop (useful
for debugging, since tracebacks from worker processes are harder to read).
"""
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src import config as C


def _simulate_one(seed):
    """Simulate exactly one (theta, noisy_image) pair. Takes an integer seed
    (not a shared rng) because each call may run in a *different process* --
    a shared np.random.Generator object cannot be passed across a process
    boundary safely, but an int can. Imports lenstronomy lazily so worker
    processes only pay that import cost once, on first use."""
    from src.simulator.parameters import sample_prior
    from src.simulator.lens_generator import simulate_clean
    from src.simulator.noise import add_noise, peak_snr

    rng = np.random.default_rng(seed)
    theta = sample_prior(rng)
    clean = simulate_clean(theta)
    noisy = add_noise(clean, rng=rng)
    theta_row = np.array([theta[name] for name in C.PARAM_NAMES], dtype=np.float32)
    return theta_row, noisy.astype(np.float32), peak_snr(clean)


def _generate(n, base_seed, report_snr=False):
    """Simulate n systems, optionally in parallel. Returns
    (theta[n, NUM_PARAMS], images[n, NUM_PIX, NUM_PIX, 1])."""
    seeds = (base_seed + np.arange(n)).tolist()  # one distinct seed per sample
    theta = np.empty((n, C.NUM_PARAMS), dtype=np.float32)
    images = np.empty((n, C.NUM_PIX, C.NUM_PIX, 1), dtype=np.float32)
    snrs = []

    t0 = time.time()
    if C.N_WORKERS > 1:
        with ProcessPoolExecutor(max_workers=C.N_WORKERS) as pool:
            for i, (theta_row, img, snr) in enumerate(pool.map(_simulate_one, seeds, chunksize=32)):
                theta[i] = theta_row
                images[i, :, :, 0] = img
                if report_snr:
                    snrs.append(snr)
                if (i + 1) % 500 == 0:
                    rate = (i + 1) / (time.time() - t0)
                    print(f"  {i + 1:>6}/{n}  ({rate:5.1f} sims/s, {C.N_WORKERS} workers)")
    else:
        for i, seed in enumerate(seeds):
            theta_row, img, snr = _simulate_one(seed)
            theta[i] = theta_row
            images[i, :, :, 0] = img
            if report_snr:
                snrs.append(snr)
            if (i + 1) % 500 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i + 1:>6}/{n}  ({rate:5.1f} sims/s, single process)")

    if report_snr and snrs:
        snrs = np.array(snrs)
        print("\n  --- peak-SNR report (tune config.SOURCE_AMP / BACKGROUND_RMS) ---")
        print(f"    min={snrs.min():5.1f}  median={np.median(snrs):5.1f}  "
              f"max={snrs.max():5.1f}   target ~ 10-30\n")
    return theta, images


def build_and_save():
    """Precompute TRAIN + VAL + TEST sets and write them to config.DATA_DIR.
    This is the function main.py's `generate` subcommand calls."""
    os.makedirs(C.DATA_DIR, exist_ok=True)

    print(f"Generating TRAIN ({C.N_TRAIN}) + VAL ({C.N_VAL}) using {C.N_WORKERS} worker(s) ...")
    theta_tr, img_tr = _generate(C.N_TRAIN, base_seed=C.SEED, report_snr=True)
    theta_va, img_va = _generate(C.N_VAL, base_seed=C.SEED + 1_000_000)

    train_path = os.path.join(C.DATA_DIR, C.DATA_FILE)
    np.savez_compressed(
        train_path,
        theta_train=theta_tr, images_train=img_tr,
        theta_val=theta_va, images_val=img_va,
        param_names=np.array(C.PARAM_NAMES),
    )
    print(f"Saved {train_path}")

    print(f"\nGenerating TEST ({C.N_TEST}) ...")
    theta_te, img_te = _generate(C.N_TEST, base_seed=C.SEED + 2_000_000)
    test_path = os.path.join(C.DATA_DIR, C.TEST_FILE)
    np.savez_compressed(
        test_path,
        theta_test=theta_te, images_test=img_te,
        param_names=np.array(C.PARAM_NAMES),
    )
    print(f"Saved {test_path}")
    print("\nDone. Next: `python main.py train`")
