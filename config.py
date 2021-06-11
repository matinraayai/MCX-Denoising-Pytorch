"""
YACS configs for training and inference
Borrowed from https://github.com/zudi-lin/pytorch_connectomics/
"""
import os
from os import cpu_count
from yacs.config import CfgNode


def get_default_training_cfg():
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

    _C.model.noise_map = False

    # DnCNN-specific arguments:
    _C.model.DnCNN = CfgNode()

    _C.model.DnCNN.do_3d = False

    _C.model.DnCNN.num_layers = 17

    _C.model.DnCNN.activation_fn = 'F.relu'

    _C.model.DnCNN.kernel_size = 3

    _C.model.DnCNN.inter_kernel_channel = 64

    # UNet-specific arguments:
    _C.model.UNet = CfgNode()

    _C.model.UNet.do_3d = False

    # Residual DnCNN-specific arguments:
    _C.model.ResidualDnCNN = CfgNode()

    _C.model.ResidualDnCNN.do_3d = False

    _C.model.ResidualDnCNN.num_layers = 17

    _C.model.ResidualDnCNN.activation_fn = 'F.relu'

    _C.model.ResidualDnCNN.kernel_size = 3

    _C.model.ResidualDnCNN.inter_kernel_channel = 64

    _C.model.ResidualDnCNN.padding_mode = 'reflect'

    # Cascaded DnCNN + UNet Specific arguments:
    _C.model.Cascaded = CfgNode()

    _C.model.Cascaded.do_3d = False

    _C.model.Cascaded.num_dncnn = 1

    _C.model.Cascaded.num_dncnn_layers = 17

    _C.model.Cascaded.activation_fn = 'F.relu'

    # DRUNet Specific arguments:
    _C.model.DRUNet = CfgNode()

    _C.model.DRUNet.num_res_blocks = 4

    _C.model.DRUNet.res_block_channels = None

    _C.model.DRUNet.activation_function = 'F.relu'

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

    _C.loss.ssim.dim = 2

    _C.loss.wmse = CfgNode()

    _C.loss.wmse.weight = 40

    _C.loss.wmse.threshold = 0.01

    _C.loss.size_average = True

    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    _C.dataset = CfgNode()

    _C.dataset.train_path = 'data/rand2d/train/'

    _C.dataset.valid_path = 'data/rand2d/validation/'

    _C.dataset.test_path = 'data/rand2d/test/'

    _C.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']

    _C.dataset.output_label = 'x1e9'

    _C.dataset.dataloader_workers = cpu_count() - 1

    _C.dataset.crop_size = None

    _C.aug = CfgNode()

    _C.aug.rotate = CfgNode({'enabled': True})

    _C.aug.rotate.p = 0.7

    _C.aug.flip = CfgNode({'enabled': True})

    _C.aug.flip.p = 1.

    # -----------------------------------------------------------------------------
    # Solver
    # -----------------------------------------------------------------------------
    _C.solver = CfgNode()

    _C.solver.batch_size = 30

    _C.solver.total_iterations = 40000
    # Specify the learning rate scheduler.
    _C.solver.lr_scheduler_name = "MultiStepLR"

    # Save a checkpoint after every this number of iterations.
    _C.solver.iteration_save = 1

    # Whether or not to restart training from iteration 0 regardless
    # of the 'iteration' key in the checkpoint file. This option only
    # works when a pretrained checkpoint is loaded (default: False).
    _C.solver.iteration_restart = False

    _C.solver.base_lr = 0.0001

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
    _C.visualize = True
    _C.experiment_name = ''
    # # -----------------------------------------------------------------------------
    # # Inference
    # # -----------------------------------------------------------------------------
    _C.inference = CfgNode()
    #
    # _C.INFERENCE.INPUT_SIZE = []
    # _C.INFERENCE.OUTPUT_SIZE = []
    #
    # _C.INFERENCE.INPUT_PATH = ""
    # _C.INFERENCE.IMAGE_NAME = ""
    _C.inference.output_dir = "./results/"

    _C.inference.checkpoint_dir = ''

    _C.inference.denoise_3d_with_2d = True

    return _C


def get_default_analysis_cfg():
    """
    Creates a YACS CfgNode with all the default options needed to analyze filtering results.
    :return: a YACS.CfgNode with all the default options for analysis
    """
    _C = CfgNode()
    # If 3D or 2D analysis
    _C.do_3d = True
    # Dataset
    _C.dataset = CfgNode()
    _C.dataset.paths = CfgNode()
    _C.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']
    _C.dataset.output_label = 'x1e9'
    # Cross section coordinates
    _C.cross_section = CfgNode()
    _C.cross_section.x = 50
    _C.cross_section.y = 50
    # Figure options
    _C.figures = CfgNode()
    _C.figures.fig_type = "display only"
    _C.output_path = "."
    # Zeroing options
    _C.zero_nans = False
    _C.zero_infs = False

    return _C


def save_all_cfg(cfg, output_dir):
    """Save configs in the output directory."""
    # Save config.yaml in the experiment directory after combine all
    # non-default configurations from yaml file and command line.
    path = os.path.join(output_dir, "config.yaml")
    with open(path, "w") as f:
        f.write(cfg.dump())
    print(f"Full config saved to {path}")


def read_training_cfg_file(config_file_path):
    cfg = get_default_training_cfg()
    cfg.update()
    cfg.merge_from_file(config_file_path)
    # Logic to switch to 2D/3D loss function for SSIM
    model_architecture = cfg.model.architecture
    cfg.loss.ssim.dim = 3 if getattr(cfg.model, model_architecture).do_3d else 2
    cfg.freeze()
    return cfg


def read_analysis_cfg_file(config_file_path):
    cfg = get_default_analysis_cfg()
    cfg.update()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg
