from torchvision.transforms import Compose
from .augmentor import DataAugment

# augmentation methods
from .flip import Flip
from .rotation import Rotate

__all__ = ['Compose',
           'DataAugment',
           'Rotate',
           'Flip']


def build_train_augmentor(**kwargs):
    aug_list = []

    # 1. rotate
    if kwargs['rotate']['enabled']:
        aug_list.append(Rotate(p=kwargs['rotate']['p']))

    # 2. flip
    if kwargs['flip']['enabled']:
        aug_list.append(Flip(p=kwargs['flip']['p']))

    return Compose(transforms=aug_list)
