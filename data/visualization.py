import numpy as np
from typing import Optional, Iterable


def get_pretty_photon_level(label: str) -> str:
    """
    Creates a latex-formatted string of the given label for the scientific notation of the number of photons.
    Each label indicates the number of photons each simulation has run with in the scientific e notation,
    and has a starting letter 'x' to make it save as a variable in MATLab. For example, a simulation that has
    10^5 photons will have the label 'x1e5'. This function converts the scientific e notation used in the MAT files
    to create a latex-formatted string to be used for visualizations inside the paper.
    :param label: the number of photons for a simulation with the scientific e notation and a redundant x in the
    beginning e.g. x1e5
    :return: Latex-formatted string, containing the number of photons written in scientific notation
    """
    return f'${label[1]}0^{label[-1]}$'


def norm_fluence_map(f_map: np.ndarray, do_clip: bool = True, clip_value: int = -64) -> np.ndarray:
    """
    Normalizes a fluence map with a high dynamic range by first taking its absolute value and then taking its log10
    for visualization. This is different from the norm function used in data/dataset.py, which uses log1p for
    normalizing the fluence maps for training.
    Since taking the log results in voxels with -inf values that show up as white space in plots,
    they are by default removed by replacing them with a small clipping value, by default set to -64. This is
    optional and can be turned off.
    :param f_map: fluence map obtained from a MAT file, simulated by MCXLab
    :param do_clip: whether to clip -inf values, default is True
    :param clip_value: the clipping value that replaces the -inf values, default is -64
    :return: fluence map with normalization applied (log10 of its absolute value) and optional clipping
    """
    f_map = np.log10(np.abs(f_map.astype(np.double))).squeeze()
    if do_clip:
        f_map[f_map == float('-inf')] = clip_value
    return f_map


def zero_contour(f_map, contour):
    """
    Zeros voxels of the fluence map that has the label 0 in the contour. These voxels fall outside of the area of
    interest and we do not want them to show up in the final visualization.
    :param f_map: fluence map as an np.ndarray
    :param contour: contour volume as an np.ndarray with the same shape as the fluence map
    :return: a new fluence map with zero-labeled voxels zeroed out in the fluence map
    """
    assert contour.shape == f_map.shape
    if contour is not None:
        f_map = np.where(contour == 0, np.zeros_like(f_map), f_map)
    return f_map


def crop_and_rotate(vol: np.ndarray, crop: Optional[Iterable[slice]] = None,
                    rotate: bool = False, k: int = 1) -> np.ndarray:
    """
    Crops and rotates the volume using np.rot90 for visualization. The volume can be both a contour or a fluence map.
    :param vol: input volume
    :param crop: If None and the input volume is 2D, does not perform any cropping. If None and the input volume is
    3D, the middle cross section
    If not, is a list of slices over each dimension for cropping
    :param rotate: whether to perform 90 degrees rotation
    :param k: the axis over the rotation is performed
    :return: the output volume, rotated and cropped
    """
    if rotate:
        vol = np.rot90(vol, k)
    if crop is None:
        if len(vol.shape) == 3:
            crop = vol.shape[0] // 2
        else:
            crop = slice(0, None)
    if len(vol.shape) == 2:
        crop = [c for c in crop if not isinstance(c, int)]
    vol = vol[crop].squeeze()
    return vol
