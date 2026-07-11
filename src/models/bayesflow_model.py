"""
src/models/bayesflow_model.py
===============================
NEW CONTENT (file previously existed but was empty). WHAT: defines the two
neural networks BayesFlow needs to turn an image into a posterior over
theta:

  1. A SUMMARY network -- a small CNN that compresses the 64x64x1 image into
     a learned SUMMARY_DIM-vector of "informative features". BayesFlow's
     built-in summary networks are designed for sets/time-series, not 2-D
     images, so for image data you must supply your own -- this is that
     network.

  2. An INFERENCE network -- a conditional normalizing flow (a "coupling
     flow") that learns p(theta | summary). Given the image summary vector,
     it maps a simple base distribution (a Gaussian) to the posterior over
     theta, so that after training we can draw posterior samples for any new
     image essentially instantly (no MCMC).

BayesFlow 2.x is built on Keras 3, so the CNN below is plain Keras and works
under any Keras backend (this project defaults to "torch", see
src/config.py / src/models/train.py).

WHY A CUSTOM CNN, NOT A BIGGER PRETRAINED NETWORK: our images are small
(64x64, single channel, simulated -- not natural photos), and we are
training on a CPU-only laptop with a training set of only a few thousand
images. A small from-scratch CNN trains in minutes and is much less prone to
overfitting on this little data than a large pretrained backbone would be.
"""
import keras
from keras import layers
import bayesflow as bf

from src import config as C


@keras.saving.register_keras_serializable()
class LensSummaryNet(bf.networks.SummaryNetwork):
    """CNN that maps (batch, H, W, 1) lens images to (batch, SUMMARY_DIM)
    feature vectors.

    Structure: three conv blocks (each: 2 conv layers + max-pool downsample +
    BatchNorm), then global average pooling and two dense layers. Each pool
    halves the spatial size (64 -> 32 -> 16 -> 8), so by the final block the
    network has seen the whole image at increasingly coarse scales -- coarse
    features capture the overall ring/arc shape, fine features (early
    layers) capture arc thickness and small asymmetries.

    BatchNorm makes the network fairly insensitive to the exact input
    pixel-value scaling, so the simple Rescaling layer below (dividing by
    IMAGE_SCALE) is all the input normalization we need.
    """

    def __init__(self, summary_dim=C.SUMMARY_DIM, **kwargs):
        super().__init__(**kwargs)
        self.summary_dim = summary_dim
        self.net = keras.Sequential([
            layers.Rescaling(1.0 / C.IMAGE_SCALE),                 # ~O(1) inputs

            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),                                 # 64 -> 32
            layers.BatchNormalization(),

            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),                                 # 32 -> 16
            layers.BatchNormalization(),

            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),                                 # 16 -> 8
            layers.BatchNormalization(),

            layers.GlobalAveragePooling2D(),                       # -> (batch, 128)
            layers.Dense(128, activation="relu"),
            layers.Dense(summary_dim),                             # -> (batch, D)
        ], name="lens_cnn")

    def call(self, x, training=False, **kwargs):
        return self.net(x, training=training)


def build_summary_network():
    # NOTE: summary_dim is passed explicitly (rather than relying on
    # LensSummaryNet's default argument) so that changing config.SUMMARY_DIM
    # from a notebook *after* this module has already been imported still
    # takes effect -- Python default arguments are bound once, at class
    # definition time, which would otherwise silently ignore later edits.
    return LensSummaryNet(summary_dim=C.SUMMARY_DIM)


def build_inference_network():
    """A coupling-flow normalizing flow -- a robust, well-tested default
    posterior network for BayesFlow.

    TRY THIS: `bf.networks.CouplingFlow` can be swapped for
    `bf.networks.FlowMatching()` or `bf.networks.DiffusionModel()` here
    without changing anything else in the pipeline (train.py just calls this
    function). Those alternatives can give slightly better posteriors but
    typically need more training steps/time -- a bigger ask on a CPU-only
    machine, so CouplingFlow is the default.
    """
    return bf.networks.CouplingFlow(depth=C.COUPLING_DEPTH)
