from .composition import Compose
from .augmentor import DataAugment

# augmentation methods
from .flip import Flip
from .rotation import Rotate

__all__ = ['Compose',
           'DataAugment',
           'Rotate',
           'Flip']


def build_train_augmentor(model_input_size, keep_uncropped=False, keep_non_smoothed=False, **kwargs):
    # The two arguments, keep_uncropped and keep_non_smoothed, are used only
    # for debugging, which are False by defaults and can not be adjusted
    # in the config files.
    aug_list = []

    # 1. rotate
    if kwargs['rotate']['enabled']:
        aug_list.append(Rotate(p=kwargs['rotate']['p']))

    # 2. flip
    if kwargs['flip']['enabled']:
        aug_list.append(Flip(p=kwargs['flip']['p']))

    augmentor = Compose(transforms=aug_list, 
                        input_size=model_input_size,
                        smooth=kwargs['smooth'],
                        keep_uncropped=keep_uncropped,
                        keep_non_smoothed=keep_non_smoothed)
    return augmentor
