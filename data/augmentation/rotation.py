from __future__ import print_function, division

import numpy as np
from .augmentor import DataAugment
from torchvision.transforms.functional import rotate
import torch


def select_random_dims(random_state):
    axes = [(1, 2), (1, 3), (2, 3)]
    return axes[random_state.randint(3)]


class Rotate(DataAugment):
    """
    Continuous rotatation of the `xy`-plane.

    If the rotation degree is arbitrary, the sample size for `x`- and `y`-axes should be at
    least :math:`\sqrt{2}` times larger than the input size to ensure there is no non-valid region
    after center-crop. This augmentation is applied to both images and masks.

    Args:
        rot90 (bool): rotate the sample by only 90 degrees. Default: True
        p (float): probability of applying the augmentation. Default: 0.5
        additional_targets(dict, optional): additional targets to augment. Default: None
    """

    def __init__(self,
                 p: float = 0.5):

        super(Rotate, self).__init__(p, None)

    def set_params(self):
        pass

    def __call__(self, sample, random_state=np.random.RandomState()):
        image, label = sample['image'], sample['label']
        if len(image.shape) == 4:
            dims = select_random_dims(random_state)
            image, label = torch.rot90(image, 1, dims), torch.rot90(label, 1, dims)
        angle = random_state.rand() * 360.0
        image = rotate(image, angle)
        label = rotate(label, angle)
        if len(image.shape) == 4:
            image, label = torch.rot90(image, -1, dims), torch.rot90(label, -1, dims)
            # An independent rotation along a plane
            k = random_state.randint(0, 3)
            dims = select_random_dims(random_state)
            image, label = torch.rot90(image, k, dims), torch.rot90(label, k, dims)
        sample['image'], sample['label'] = image, label
        return sample
