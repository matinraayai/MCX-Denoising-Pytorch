import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from evaluation.utils import read_mat_files, prepare_label_to_idx_mapping_for_analysis
from config import read_vis_cfg_file
import scipy.io as sio


def get_args():
    parser = argparse.ArgumentParser(description="Script for visualizing filtering results for both 2D and 3D real"
                                                 "datasets.")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). '
                                                        'Refer to config.py and the config/analysis directory'
                                                        'for more info on the format.')
    return parser.parse_args()


def get_pretty_photon_level(input_label):
    return f'${input_label[1]}0^{input_label[-1]}$'


def crop_contour(contour, crop=None, rotate=False, k=1):
    if len(contour.shape) == 3:
        if crop is None:
            crop = contour.shape[0] // 2
        contour = contour[crop].squeeze()
    if rotate:
        contour = np.rot90(contour, k)
    return contour


def normalize_fluence_map_and_crop(f_map, crop=None, rotate=False, k=1):
    f_map = np.log1p(f_map.astype(np.double)).squeeze()

    if crop is None:
        if len(f_map.shape) == 3:
            crop = f_map.shape[0] // 2
        else:
            crop = slice(0, None)
    f_map = f_map[crop].squeeze()
    if rotate:
        f_map = np.rot90(f_map, k)
    return f_map


def plot_and_add_colorbar(ax, img, cmap, font, contour):
    x = np.linspace(0, img.shape[1], img.shape[1])
    y = np.linspace(0, img.shape[0], img.shape[0])
    x, y = np.meshgrid(x, y)
    ax.contour(x, y, contour, colors='black')
    return ax.imshow(img, cmap=cmap)


def plot(datasets, contour, input_labels, output_label, crop, rotate, k_rotate, plot_output_dir,
         dataset_name_on_rows=False, size=60, fig_size=(50, 50),
         cmap='jet'):
    """
    model -> label -> sample
    """
    num_models = len(datasets) + 1
    num_labels = len(input_labels)
    os.makedirs(plot_output_dir, exist_ok=True)
    font = {'family': 'times new roman', 'color': 'black', 'size': size}
    fig, axs = plt.subplots(num_labels, num_models, figsize=fig_size, subplot_kw={'xticks': [], 'yticks': []},
                            constrained_layout=True)
    contour = crop_contour(contour, crop, rotate, k_rotate)
    # Label Setup
    # Place input label's column first
    print(f"Input Labels and Ground Truth")

    axs[0, 0].set_title('Input Simulation', font)
    # Place output label's column last
    gt_column_label = get_pretty_photon_level(output_label)
    axs[0, -1].set_title(gt_column_label, font)
    # Do this for the rest of the models
    j = 1
    for model_name in datasets.keys():
        if model_name != 'simulation':
            print(f"Dataset: {model_name}")
            axs[0, j].set_title(model_name, font)
            j += 1

    for j, input_label in enumerate(sorted(input_labels)):
        if dataset_name_on_rows:
            input_column_label = get_pretty_photon_level(input_label)
            axs[j, 0].set_ylabel(input_column_label, font)
        input_sample = datasets['simulation'][input_label]
        gt_sample = datasets['simulation'][output_label]
        input_norm_sample = normalize_fluence_map_and_crop(input_sample, crop, rotate, k_rotate)
        gt_norm_sample = normalize_fluence_map_and_crop(gt_sample, crop, rotate, k_rotate)
        im = plot_and_add_colorbar(axs[j, 0], input_norm_sample, cmap, font, contour)
        im = plot_and_add_colorbar(axs[j, -1], gt_norm_sample, cmap, font, contour)
        # Do the rest of the datasets
        i = 1
        for model_name in datasets.keys():
            if model_name != 'simulation':
                sample = datasets[model_name][input_label]
                norm_sample = normalize_fluence_map_and_crop(sample, crop, rotate, k_rotate)
                im = plot_and_add_colorbar(axs[j, i], norm_sample, cmap, font, contour)
                i += 1

    plt.subplots_adjust(hspace=0.01, wspace=0.05)
    cbar = plt.colorbar(im, ax=axs.ravel().tolist())
    cbar.ax.tick_params(labelsize=size, direction='out', length=size//3)
    fig.savefig(os.path.join(plot_output_dir, "visualization.pdf"))
    plt.close(fig)


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = read_vis_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)

    # Read in all the datasets. The datasets dictionary will have the following mapping:
    # model_name -> label (e.g. 'x1e5') -> fluence map (e.g. np.ndarray(100, 100, 100))
    datasets = {}
    for model_name, path in cfg.dataset.paths.items():
        datasets[model_name] = {}
        mat_files = read_mat_files(path, max_num_files=1)
        datasets[model_name] = prepare_label_to_idx_mapping_for_analysis(mat_files)
    contour = sio.loadmat(cfg.contour.path)[cfg.contour.name]
    print("Plotting Vis.")
    crop = None if cfg.dataset.crop is None else eval(cfg.dataset.crop)
    plot(datasets, contour,
         cfg.dataset.input_labels, cfg.dataset.output_label, crop, cfg.dataset.rotate.do, cfg.dataset.rotate.k,
         cfg.output_dir,
         cfg.dataset_name_on_rows, cfg.font_size, cfg.fig_size)
    print("Done plotting.")


if __name__ == '__main__':
    main()
