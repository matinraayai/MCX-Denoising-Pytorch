"""
YACS configs for training and inference
Borrowed from https://github.com/zudi-lin/pytorch_connectomics/
"""
import os
from os import cpu_count

import torch.cuda
from yacs.config import CfgNode


def add_model_cfg(cfg: CfgNode):
    """
    Adds model arguments to a YACS CfgNode object
    :param cfg: Config node to be used for a specific task
    :return: The same config node, with model config node added
    """
    cfg.model = CfgNode()

    cfg.model.checkpoint = None

    cfg.model.architecture = 'DnCNN'

    # DnCNN-specific arguments:
    cfg.model.DnCNN = CfgNode()

    cfg.model.DnCNN.do_3d = False

    cfg.model.DnCNN.num_layers = 17

    cfg.model.DnCNN.activation_fn = 'F.relu'

    cfg.model.DnCNN.padding_mode = 'reflect'

    cfg.model.DnCNN.kernel_size = 3

    cfg.model.DnCNN.inter_kernel_channel = 64

    cfg.model.DnCNN.init_policy = None

    # UNet-specific arguments:
    cfg.model.UNet = CfgNode()

    cfg.model.UNet.do_3d = False

    cfg.model.UNet.activation_fn = 'F.relu'

    cfg.model.UNet.init_policy = None

    # Residual DnCNN-specific arguments:
    cfg.model.ResidualDnCNN = CfgNode()

    cfg.model.ResidualDnCNN.do_3d = False

    cfg.model.ResidualDnCNN.num_layers = 17

    cfg.model.ResidualDnCNN.activation_fn = 'F.relu'

    cfg.model.ResidualDnCNN.kernel_size = 3

    cfg.model.ResidualDnCNN.inter_kernel_channel = 64

    cfg.model.ResidualDnCNN.padding_mode = 'reflect'

    cfg.model.ResidualDnCNN.init_policy = None

    # Cascaded DnCNN + UNet Specific arguments:
    cfg.model.Cascaded = CfgNode()

    cfg.model.Cascaded.do_3d = False

    cfg.model.Cascaded.num_dncnn_layers = 17

    cfg.model.Cascaded.dncnn_activation_fn = 'F.relu'

    cfg.model.Cascaded.unet_activation_fn = 'F.relu'

    cfg.model.Cascaded.padding_mode = 'reflect'

    cfg.model.Cascaded.init_policy = None

    # DRUNet Specific arguments:
    cfg.model.DRUNet = CfgNode()

    cfg.model.DRUNet.do_3d = False

    cfg.model.DRUNet.num_res_blocks = 4

    cfg.model.DRUNet.res_block_channels = None

    cfg.model.DRUNet.activation_function = 'F.relu'

    cfg.model.DRUNet.init_policy = None

    return cfg


def add_pytorch_lightning_cfg(cfg: CfgNode):
    """
    Adds pytorch_lightning specific arguments to a configuration node under the name "pytorch_lightning"
    :param cfg: Config node to be used for a specific task
    :return: The same config node, with pytorch_lightning node added
    """
    cfg.pytorch_lightning = CfgNode()
    cfg.pytorch_lightning.num_gpus = torch.cuda.device_count()

    cfg.pytorch_lightning.num_nodes = 1

    cfg.pytorch_lightning.checkpoint_dir = ''

    cfg.pytorch_lightning.visualize = False

    cfg.pytorch_lightning.experiment_name = ''

    cfg.pytorch_lightning.accelerator = None

    cfg.pytorch_lightning.resume_training_checkpoint = None

    # If True, sets every seed in the training to 1 for reproducibility
    cfg.pytorch_lightning.seed_everything = True

    # Seed to use if seed_everything is set to True
    cfg.pytorch_lightning.seed = 1

    return cfg


def add_loss_cfg(cfg: CfgNode):
    """
    Adds loss function arguments to a configuration node under the name "loss"
    :param cfg: Config node to be used for a specific task
    :return: The same config node, with loss node added
    """
    cfg.loss = CfgNode()

    cfg.loss.loss_option = ('MSE',)

    cfg.loss.loss_weight = (1.0,)

    cfg.loss.regularizer_opt = ()

    cfg.loss.regularizer_weight = ()

    # SSIM specific arguments
    cfg.loss.ssim = CfgNode()

    cfg.loss.ssim.window_size = 11

    cfg.loss.ssim.dim = 2

    cfg.loss.wmse = CfgNode()

    cfg.loss.wmse.weight = 40

    cfg.loss.wmse.threshold = 0.01

    cfg.loss.size_average = True
    return cfg


def add_dataset_cfg(cfg: CfgNode, num_gpus):
    """
    Adds dataset function arguments to a configuration node under the name "dataset"
    :param cfg: Config node to be used for a specific task
    :param num_gpus: Number of GPUs to be used for the task; Used to set the default argument of the dataloader_workers
    :return: The same config node, with loss node added
    """
    cfg.dataset = CfgNode()

    cfg.dataset.train_path = 'data/train/3D'

    cfg.dataset.valid_path = 'data/validation/3D'

    cfg.dataset.test_path = 'data/test/3D'

    cfg.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']

    cfg.dataset.output_label = 'x1e9'

    cfg.dataset.valid_labels = ['x1e5']

    cfg.dataset.dataloader_workers = cpu_count() // num_gpus

    cfg.dataset.crop_size = None

    cfg.dataset.padding = None
    return cfg


def add_aug_cfg(cfg: CfgNode):
    """
    Adds augmentation function arguments to a configuration node under the name "aug"
    :param cfg: Config node to be used for a specific task
    :return: The same config node, with aug node added
    """
    cfg.aug = CfgNode()

    cfg.aug.rotate = CfgNode({'enabled': True})

    cfg.aug.rotate.p = 0.7

    cfg.aug.flip = CfgNode({'enabled': True})

    cfg.aug.flip.p = 1.

    return cfg


def add_solver_cfg(cfg: CfgNode):
    """
    Adds solver function arguments to a configuration node under the name "solver"
    :param cfg: Config node to be used for a specific task
    :return: The same config node, with solver node added
    """
    cfg.solver = CfgNode()

    cfg.solver.optimizer = 'SGD'

    cfg.solver.batch_size = 32

    cfg.solver.total_iterations = 1000
    # Specify the learning rate scheduler.
    cfg.solver.lr_scheduler_name = "MultiStepLR"

    # Whether or not to restart training from iteration 0 regardless
    # of the 'iteration' key in the checkpoint file. This option only
    # works when a pretrained checkpoint is loaded (default: False).
    cfg.solver.iteration_restart = False

    cfg.solver.base_lr = 0.0001

    cfg.solver.bias_lr_factor = 1.0

    cfg.solver.weight_decay_bias = 0.0

    cfg.solver.momentum = 0.9

    # The weight decay that's applied to parameters of normalization layers
    # (typically the affine transformation)
    cfg.solver.weight_decay = 0.0001

    cfg.solver.weight_decay_norm = 0.0

    # The iteration number to decrease learning rate by GAMMA
    cfg.solver.gamma = 0.1

    # should be a tuple like (30000,)
    cfg.solver.steps = (600, 800)

    cfg.solver.warmup_factor = 1.0 / 1000

    cfg.solver.warmup_iters = 2

    cfg.solver.warmup_method = 'linear'

    # Gradient clipping
    cfg.solver.clip_gradients = CfgNode({"enabled": False})
    # Type of gradient clipping, currently 2 values are supported:
    # - "value": the absolute values of elements of each gradients are clipped
    # - "norm": the norm of the gradient for each parameter is clipped thus
    #   affecting all elements in the parameter
    cfg.solver.clip_gradients.clip_type = "value"
    # Maximum absolute value used for clipping gradients
    cfg.solver.clip_gradients.clip_value = 1.0
    # Floating point number p for L-p norm to be used with the "norm"
    # gradient clipping type; for L-inf, please specify .inf
    cfg.solver.clip_gradients.norm_type = 2.0
    return cfg


def get_default_training_cfg():
    _C = CfgNode()
    _C = add_pytorch_lightning_cfg(_C)
    _C = add_model_cfg(_C)
    _C = add_loss_cfg(_C)
    _C = add_dataset_cfg(_C, _C.pytorch_lightning.num_gpus)
    _C = add_aug_cfg(_C)
    _C = add_solver_cfg(_C)
    return _C


def get_default_inference_cfg():
    _C = CfgNode()
    _C = add_model_cfg(_C)
    _C = add_dataset_cfg(_C, 1)
    _C = add_loss_cfg(_C)
    _C.output_dir = "./results"
    return _C


def get_default_cross_section_analysis_cfg():
    """
    Creates a YACS CfgNode with all the default options needed to analyze filtering results.
    :return: a YACS.CfgNode with all the default options for analysis
    """
    _C = CfgNode()
    # Dataset
    _C.dataset = CfgNode()
    _C.dataset.paths = CfgNode()
    _C.dataset.labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8', 'x1e9']
    # Cross section coordinates
    _C.cross_section = CfgNode()
    _C.cross_section.x = 50
    _C.cross_section.y = None
    # Figure options
    _C.figures = CfgNode()
    _C.figures.legend = True
    _C.figures.fig_type = "display only"
    _C.output_path = "."
    # Zeroing options
    _C.zero_nans = False
    _C.zero_infs = False
    return _C


def get_default_global_metrics_analysis_cfg():
    """
    Creates a YACS CfgNode with all the default options needed to analyze global metrics of the filtering results
    :return: a YACS.CfgNode with all the default options for analysis
    """
    _C = CfgNode()
    # Dataset
    _C.dataset = CfgNode()
    _C.dataset.paths = CfgNode()
    _C.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']
    _C.dataset.output_label = 'x1e9'
    _C.output_path = "."
    return _C


def get_default_profiling_cfg():
    _C = CfgNode()
    _C = add_model_cfg(_C)
    _C.input_dims = (64, 64, 64)
    _C.num_iterations = 100
    _C.unpaded_volume_slice = ""
    return _C


def get_default_vis_cfg():
    _C = CfgNode()
    # Dataset
    _C.dataset = CfgNode()
    _C.dataset.paths = CfgNode()
    _C.dataset.input_labels = ['x1e5', 'x1e6', 'x1e7', 'x1e8']
    _C.dataset.output_label = 'x1e9'
    _C.output_dir = "."
    _C.dataset_name_on_rows = True
    _C.font_size = 60
    _C.fig_size = (30, 30)
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


def read_cross_section_analysis_cfg_file(config_file_path):
    cfg = get_default_cross_section_analysis_cfg()
    cfg.update()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


def read_global_metrics_analysis_cfg_file(config_file_path):
    cfg = get_default_global_metrics_analysis_cfg()
    cfg.update()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


def read_profiling_cfg_file(config_file_path):
    cfg = get_default_profiling_cfg()
    cfg.update()
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg


def read_vis_cfg_file(config_file_path):
    cfg = get_default_vis_cfg()
    cfg.update()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(config_file_path)
    cfg.freeze()
    return cfg
