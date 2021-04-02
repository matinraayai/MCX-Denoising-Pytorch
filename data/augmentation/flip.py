from __future__ import print_function, division
from typing import Optional
from .augmentor import DataAugment
import torch
import numpy as np
from torchvision.transforms.functional import vflip, hflip


def select_random_dims(random_state):
    axes = [(1, 2), (1, 3), (2, 3)]
    return axes[random_state.randint(3)]


class Flip(DataAugment):
    """
    Randomly flip along `z`-, `y`- and `x`-axes. For learning on isotropic image volumes set
    :attr:`do_ztrans` to 1 to swap `z`- and `x`-axes (the inputs need to be cubic).
    This augmentation is applied to both images and masks.

    Args:
        do_ztrans (int): set to 1 to swap z- and x-axes for isotropic data. Default: 0
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """
    def __init__(self,
                 p: float = 0.5,
                 additional_targets: Optional[dict] = None):

        super(Flip, self).__init__(p, additional_targets)

    def set_params(self):
        pass
    
    def __call__(self, sample, random_state=np.random.RandomState()):
        image, label = sample['image'], sample['label']
        if len(image.shape) == 4:
            dims = select_random_dims(random_state)
            image, label = torch.rot90(image, 1, dims), torch.rot90(label, 1, dims)
        flip_function = vflip if random_state.randint(1) else hflip
        image, label = flip_function(image), flip_function(label)
        if len(image.shape) == 4:
            image, label = torch.rot90(image, -1, dims), torch.rot90(label, -1, dims)
        sample['image'], sample['label'] = image, label
        return sample
