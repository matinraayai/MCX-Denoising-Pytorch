import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from evaluation.utils import read_mat_files, prepare_label_to_idx_mapping_for_analysis
from config import read_vis_cfg_file


def get_args():
    parser = argparse.ArgumentParser(description="Script for visualizing filtering results for both 2D and 3D.")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). '
                                                        'Refer to config.py and the config/analysis directory'
                                                        'for more info on the format.')
    return parser.parse_args()


def get_pretty_photon_level(input_label):
    return f'${input_label[1]}0^{input_label[-1]}$'


def normalize_fluence_map_and_crop(f_map):
    f_map = np.log1p(f_map.astype(np.double)).squeeze()
    if len(f_map.shape) == 3:
        f_map = f_map[f_map.shape[0] // 2]
    return f_map


def plot_and_add_colorbar(ax, img, cmap, font):
    return ax.imshow(img, cmap=cmap)


def plot(datasets, input_labels, output_label, plot_output_dir, dataset_name_on_rows=False, size=60, fig_size=(50, 50),
         cmap='jet'):
    """
    model -> dataset name -> label -> sample
    """
    num_datasets = len(datasets) + 1
    num_samples = len(datasets['simulation'])
    os.makedirs(plot_output_dir, exist_ok=True)
    font = {'family': 'serif', 'color': 'black', 'size': size}
    for label in input_labels:
        print(f"Generating viz for {label}")
        fig, axs = plt.subplots(num_samples, num_datasets, figsize=fig_size, subplot_kw={'xticks': [], 'yticks': []},
                                constrained_layout=True)
        # Place input label's column first
        print(f"Input Labels and Ground Truth")
        input_column_label = get_pretty_photon_level(label)
        axs[0, 0].set_title(f"Input Simulation ({input_column_label})", font)
        # Place output label's column last
        gt_column_label = get_pretty_photon_level(output_label)
        axs[0, -1].set_title(f"{gt_column_label} (Ground Truth)", font)
        for j, dataset_name in enumerate(sorted(datasets['simulation'].keys())):
            if dataset_name_on_rows:
                axs[j, 0].set_ylabel(dataset_name, font)
            input_sample = datasets['simulation'][dataset_name][label]
            gt_sample = datasets['simulation'][dataset_name][output_label]
            input_norm_sample, gt_norm_sample = normalize_fluence_map_and_crop(input_sample), \
                                                normalize_fluence_map_and_crop(gt_sample)
            im = plot_and_add_colorbar(axs[j, 0], input_norm_sample, cmap, font)
            im = plot_and_add_colorbar(axs[j, -1], gt_norm_sample, cmap, font)
        # Do the rest of the datasets
        i = 1
        for model_name, dataset in datasets.items():
            if model_name != 'simulation':
                print(f"Dataset: {model_name}")
                axs[0, i].set_title(model_name, {'family': 'serif', 'color': 'black', 'size': size})
                for j, dataset_name in enumerate(sorted(dataset.keys())):
                    sample = dataset[dataset_name][label]
                    norm_sample = normalize_fluence_map_and_crop(sample)
                    im = plot_and_add_colorbar(axs[j, i], norm_sample, cmap, font)
                i += 1

        plt.subplots_adjust(hspace=0.05, wspace=0.05)
        cbar = plt.colorbar(im, ax=axs.ravel().tolist())
        cbar.ax.tick_params(labelsize=size, direction='out', length=size//3)
        fig.savefig(os.path.join(plot_output_dir, f"{label}.pdf"))
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
    # dataset_name (e.g. simulation) -> label (e.g. 'x1e5') -> fluence map (e.g. np.ndarray(100, 100, 100))
    datasets = {}
    for model_name, dataset in cfg.dataset.paths.items():
        datasets[model_name] = {}
        for dataset_name in dataset.keys():
            mat_files = read_mat_files(dataset[dataset_name], max_num_files=1)
            datasets[model_name][dataset_name] = prepare_label_to_idx_mapping_for_analysis(mat_files)
    print("Plotting Vis.")
    plot(datasets, cfg.dataset.input_labels, cfg.dataset.output_label, cfg.output_path, cfg.dataset_name_on_rows,
         cfg.font_size, cfg.fig_size)
    print("Done plotting.")


if __name__ == '__main__':
    main()
