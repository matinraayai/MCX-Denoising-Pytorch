# Code adapted from Detectron2 (https://github.com/facebookresearch/detectron2)
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Set, Type, Union
import torch

from yacs.config import CfgNode
from .lars import LARS

from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, ExponentialLR
from .lr_scheduler import WarmupCosineLR, WarmupMultiStepLR

# 0. Gradient Clipping

_GradientClipperInput = Union[torch.Tensor, Iterable[torch.Tensor]]
_GradientClipper = Callable[[_GradientClipperInput], None]


class GradientClipType(Enum):
    VALUE = "value"
    NORM = "norm"


def _create_gradient_clipper(cfg: CfgNode) -> _GradientClipper:
    """
    Creates gradient clipping closure to clip by value or by norm,
    according to the provided config.
    """
    cfg = cfg.clone()

    def clip_grad_norm(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_norm_(p, cfg.clip_value, cfg.norm_type)

    def clip_grad_value(p: _GradientClipperInput):
        torch.nn.utils.clip_grad_value_(p, cfg.clip_value)

    _GRADIENT_CLIP_TYPE_TO_CLIPPER = {
        GradientClipType.VALUE: clip_grad_value,
        GradientClipType.NORM: clip_grad_norm,
    }
    return _GRADIENT_CLIP_TYPE_TO_CLIPPER[GradientClipType(cfg.clip_type)]


def _generate_optimizer_class_with_gradient_clipping(
    optimizer_type: Type[torch.optim.Optimizer], gradient_clipper: _GradientClipper
) -> Type[torch.optim.Optimizer]:
    """
    Dynamically creates a new type that inherits the type of a given instance
    and overrides the `step` method to add gradient clipping
    """

    def optimizer_wgc_step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                gradient_clipper(p)
        super(type(self), self).step(closure)

    OptimizerWithGradientClip = type(
        optimizer_type.__name__ + "WithGradientClip",
        (optimizer_type,),
        {"step": optimizer_wgc_step},
    )
    return OptimizerWithGradientClip


def maybe_add_gradient_clipping(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.Optimizer:
    """
    If gradient clipping is enabled through config options, wraps the existing
    optimizer instance of some type OptimizerType to become an instance
    of the new dynamically created class OptimizerTypeWithGradientClip
    that inherits OptimizerType and overrides the `step` method to
    include gradient clipping.
    Args:
        cfg: CfgNode
            configuration options
        optimizer: torch.optim.Optimizer
            existing optimizer instance
    Return:
        optimizer: torch.optim.Optimizer
            either the unmodified optimizer instance (if gradient clipping is
            disabled), or the same instance with adjusted __class__ to override
            the `step` method and include gradient clipping
    """
    if not cfg.solver.clip_gradients.enabled:
        return optimizer
    grad_clipper = _create_gradient_clipper(cfg.solver.clip_gradients)
    OptimizerWithGradientClip = _generate_optimizer_class_with_gradient_clipping(
        type(optimizer), grad_clipper
    )
    optimizer.__class__ = OptimizerWithGradientClip
    return optimizer

# 1. Build Optimizer


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    for module in model.modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.solver.base_lr
            weight_decay = cfg.solver.weight_decay
            if isinstance(module, norm_module_types):
                weight_decay = cfg.solver.weight_decay_norm
            elif key == "bias":
                lr = cfg.solver.base_lr * cfg.solver.bias_lr_factor
                weight_decay = cfg.solver.weight_decay_bias
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.solver.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params, cfg.solver.base_lr)
    elif cfg.solver.optimizer.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, cfg.solver.base_lr, momentum=cfg.solver.momentum)
    elif cfg.solver.optimizer.lower() == 'lars':
        optimizer = LARS(params, lr=cfg.solver.base_lr, momentum=cfg.solver.momentum,
                         max_epoch=cfg.solver.total_iterations)
    else:
        raise NotImplementedError(f"Invalid optimizer name {cfg.solver.optimizer}.")
    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    print('Optimizer: ', optimizer.__class__.__name__)
    return optimizer

# 2. Build LR Scheduler


def build_lr_scheduler(
    cfg: CfgNode, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.solver.lr_scheduler_name
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.solver.steps,
            cfg.solver.gamma,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_iters=cfg.solver.warmup_iters,
            warmup_method=cfg.solver.warmup_method,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.solver.total_iterations,
            warmup_factor=cfg.solver.warmup_factor,
            warmup_iters=cfg.solver.warmup_iters,
            warmup_method=cfg.solver.warmup_method,
        )
    elif name == "MultiStepLR":
        return MultiStepLR(
            optimizer,
            milestones=cfg.solver.steps,
            gamma=cfg.solver.gamma
        )
    elif name == "ReduceLROnPlateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min', factor=cfg.solver.gamma, patience=50,
            threshold=0.001, threshold_mode='rel', cooldown=0,
            min_lr=1e-06, eps=1e-08
        )
    elif name == "ExponentialLR":
        return ExponentialLR(optimizer, cfg.solver.gamma,
                             cfg.solver.total_iterations)
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
