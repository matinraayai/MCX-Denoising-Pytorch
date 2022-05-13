from __future__ import print_function, division

import numpy as np
from .augmentor import DataAugment
from torchvision.transforms.functional import rotate
import torch


def select_random_plane(random_state):
    """
    Selects a random plane using the Numpy random state to be used for applying a random rotation
    :param random_state: An np.random.RandomState object
    :return: a tuple of size 2, containing the axis of the randomly selected plane
    """
    axes = [(1, 2), (1, 3), (2, 3)]
    return axes[random_state.randint(3)]


class Rotate(DataAugment):
    """
    Applies a random rotation to a random plane of the input volume
    :param p: Probability of applying rotation on the image/volume
    """
    def __init__(self,
                 p: float = 0.5):

        super(Rotate, self).__init__(p, None)

    def set_params(self):
        pass

    def __call__(self, sample, random_state=np.random.RandomState()):
        image, label = sample['image'], sample['label']
        # Since rotate in torchvision only applies its rotation to the first plane, we first have to rotate it by 90
        # degrees randomly to bring the target plane forward for the rotation
        if len(image.shape) == 4:
            dims = select_random_plane(random_state)
            image, label = torch.rot90(image, 1, dims), torch.rot90(label, 1, dims)
        angle = random_state.rand() * 360.0
        image = rotate(image, angle)
        label = rotate(label, angle)
        if len(image.shape) == 4:
            image, label = torch.rot90(image, -1, dims), torch.rot90(label, -1, dims)
            # An independent rotation along a plane
            k = random_state.randint(0, 3)
            dims = select_random_plane(random_state)
            # Random 90-degree rotation to complete the random 3D rotation
            image, label = torch.rot90(image, k, dims), torch.rot90(label, k, dims)
        sample['image'], sample['label'] = image, label
        return sample
