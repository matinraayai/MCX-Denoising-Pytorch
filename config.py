"""
YACS configs for training and inference
Borrowed from https://github.com/zudi-lin/pytorch_connectomics/
"""
import os
from os import cpu_count
from yacs.config import CfgNode
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.model = CfgNode()

_C.model.starting_checkpoint = ''

_C.model.architecture = 'DnCNN'

# DnCNN-specific arguments:
_C.model.DnCNN = CfgNode()

_C.model.DnCNN.num_layers = 17

_C.model.DnCNN.activation_fn = 'F.relu'

_C.model.DnCNN.kernel_size = 3

_C.model.DnCNN.inter_kernel_channel = 64

# UNet-specific arguments:
_C.model.UNet = CfgNode()

# Residual DnCNN-specific arguments:
_C.model.ResidualDnCNN = CfgNode()

_C.model.ResidualDnCNN.num_layers = 17

_C.model.ResidualDnCNN.activation_fn = 'F.relu'

_C.model.ResidualDnCNN.kernel_size = 3

_C.model.ResidualDnCNN.inter_kernel_channel = 64

_C.model.ResidualDnCNN.padding_mode = 'reflect'

# -----------------------------------------------------------------------------
# Loss Options
# -----------------------------------------------------------------------------
_C.loss = CfgNode()

_C.loss.loss_option = ('MSE',)

_C.loss.loss_weight = (1.0,)

_C.loss.regularizer_opt = ()

_C.loss.regularizer_weight = ()

# SSIM specific arguments
_C.loss.ssim = CfgNode()

_C.loss.ssim.window_size = 11

_C.loss.size_average = True

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.dataset = CfgNode()

_C.dataset.train_path = './data/rand2d/'

_C.dataset.valid_path = './data/rand2d-val/'

_C.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']

_C.dataset.output_label = 'x1e9'

_C.dataset.dataloader_workers = cpu_count() - 1

_C.dataset.max_rotation_angle = 90.

_C.dataset.rotation_p = .7

_C.dataset.flip_p = .5

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.solver = CfgNode()

_C.solver.batch_size = 30

_C.solver.total_iterations = 40000
# Specify the learning rate scheduler.
_C.solver.lr_scheduler_name = "MultiStepLR"

_C.solver.iteration_step = 1

# Save a checkpoint after every this number of iterations.
_C.solver.iteration_save = 50

# Whether or not to restart training from iteration 0 regardless
# of the 'iteration' key in the checkpoint file. This option only
# works when a pretrained checkpoint is loaded (default: False).
_C.solver.iteration_restart = False

_C.solver.base_lr = 0.001

_C.solver.bias_lr_factor = 1.0

_C.solver.weight_decay_bias = 0.0

_C.solver.momentum = 0.9

# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.solver.weight_decay = 0.0001

_C.solver.weight_decay_norm = 0.0

# The iteration number to decrease learning rate by GAMMA
_C.solver.gamma = 0.1

# should be a tuple like (30000,)
_C.solver.steps = (30000, 35000)

_C.solver.warmup_factor = 1.0 / 1000

_C.solver.warmup_iters = 1000

_C.solver.warmup_method = 'linear'

# Number of samples per batch across all machines.
# If we have 16 GPUs and IMS_PER_BATCH = 32,
# each GPU will see 2 images per batch.
_C.solver.samples_per_batch = 16

# Gradient clipping
_C.solver.clip_gradients = CfgNode({"enabled": False})
# Type of gradient clipping, currently 2 values are supported:
# - "value": the absolute values of elements of each gradients are clipped
# - "norm": the norm of the gradient for each parameter is clipped thus
#   affecting all elements in the parameter
_C.solver.clip_gradients.clip_type = "value"
# Maximum absolute value used for clipping gradients
_C.solver.clip_gradients.clip_value = 1.0
# Floating point number p for L-p norm to be used with the "norm"
# gradient clipping type; for L-inf, please specify .inf
_C.solver.clip_gradients.norm_type = 2.0


_C.checkpoint_dir = ''
# # -----------------------------------------------------------------------------
# # Inference
# # -----------------------------------------------------------------------------
# _C.INFERENCE = CfgNode()
#
# _C.INFERENCE.INPUT_SIZE = []
# _C.INFERENCE.OUTPUT_SIZE = []
#
# _C.INFERENCE.INPUT_PATH = ""
# _C.INFERENCE.IMAGE_NAME = ""
# _C.INFERENCE.OUTPUT_PATH = ""
# _C.INFERENCE.OUTPUT_NAME = 'result'
#
# _C.INFERENCE.PAD_SIZE = []
#
# _C.INFERENCE.STRIDE = [4, 128, 129]
#
# # Blending function for overlapping inference.
# _C.INFERENCE.BLENDING = 'gaussian'
#
# _C.INFERENCE.AUG_MODE = 'mean'
# _C.INFERENCE.AUG_NUM = 4
#
# # Run the model forward pass with model.eval() if DO_EVAL is True, else
# # run with model.train(). Layers like batchnorm and dropout will be affected.
# _C.INFERENCE.DO_EVAL = True
#
# _C.INFERENCE.DO_3D = True
#
# # If not None then select channel of output
# _C.INFERENCE.MODEL_OUTPUT_ID = [None]
#
# # Number of test workers
# _C.INFERENCE.TEST_NUM = 1
#
# # Test worker id
# _C.INFERENCE.TEST_ID = 0
#
# # Batchsize for inference
# _C.INFERENCE.SAMPLES_PER_BATCH = 32

#######################################################
# Util functions
#######################################################


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()


def save_all_cfg(cfg, output_dir):
    """Save configs in the output directory."""
    # Save config.yaml in the experiment directory after combine all
    # non-default configurations from yaml file and command line.
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print("Full config saved to {}".format(path))
