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


def plot_stats(stat_dicts, x_cross_section, y_cross_section, labels, fig_type, output_path):
    """
    Le very complicated plotting function. Basically a wrapper around all the plotting needs this script tries to
    address, which is plotting every cross section statistics for every dataset and every label.
    As it is a very broad method it uses smaller helper functions defined in its body to do the job, each documented
    on its own.
    :param stat_dicts: A dictionary with all the previously calculated statistics using compute_cross_section_stats.
    It should have the following format:
    stat_name (e.g. snr) -> dataset_name (e.g. simulation) -> label (e.g. 'x1e5') -> fluence map
    (e.g. np.ndarray(100, 100, 100))
    Plot sizes are fixed and adjusted manually in the function.
    :param x_cross_section: X-axis cross section (used for axis labeling)
    :param y_cross_section: Y-axis cross section (used for axis labeling)
    :param labels: All the labels present in the dataset, including input labels and the output label. It is used for
    convenience, since all the labels can be extracted from the stat_dicts
    :param fig_type: Type of the figure, either "save" as an image to the file system or "display" or both.
    :param output_path: In case of "save" fig_type, path to save the figures to. Each figure will be saved in the path
    with the name of the stat e.g. "snr.png".
    :return: None
    """
    def add_label_to_stat_plot(label, stat_name, color):
        """
        Adds all the dataset cross section stats asscociated with a single label. For example, it adds the snr stats
        for label 'x1e5' for every data series which include simulation and maybe two or more denoised results using
        conventional and deep learning methods.
        It skips adding the data to the plot if it isn't present in the data, which is the case for the output label
        for CNN predictions.
        Line styles are selected from a pre-selected range of styles.
        :param label: label in the data, e.g. x1e5
        :param stat_name: name of the stat to be plot, e.g. snr, mean, or std
        :param color: Color for the lines. Each label will get assigned the same color
        :return: None
        """
        plot_label = f"$10^{int(label[-1])}$"
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'loosely dashed', 'densely dashed']
        for i, (data_name, data) in enumerate(stat_dicts.items()):
            if label in data:
                if label != 'x1e9':
                    x_values = np.arange(0, len(data[label][stat_name]))
                    y_values = data[label][stat_name]
                    plt.plot(x_values, y_values, color=color, label=f"{plot_label}_{data_name}", linestyle=linestyles[i])

    def create_stat_plot(stat_name, x_axis_label, y_axis_label, legend=False):
        """
        Creates individual plots for each stat, including mean, std, and SNR.
        :param stat_name: name of the stat
        :param x_axis_label: label for the x-axis
        :param y_axis_label: label for the y-axis
        :param legend: Plot legend is also added if True, disabled by default
        :return: None
        """
        c_map = plt.cm.get_cmap('hsv', 6)

        for i, label in enumerate(labels):
            add_label_to_stat_plot(label, stat_name, c_map(i))
        if legend:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        """These arguments are supplied by the outer function scope."""
        if "display" in fig_type:
            plt.show()
        if "save" in fig_type:
            os.makedirs(output_path, exist_ok=True)
            plt.savefig(os.path.join(output_path, f'{stat_name}.png'))
        plt.close()

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(right=0.63, bottom=0.25)
    create_stat_plot('snr', f'Y over Cross Section at $X = {x_cross_section} mm$, $Y = {y_cross_section} mm$ (mm)',
                     'SNR (DBs)', True)

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(right=0.63, bottom=0.25)
    create_stat_plot('means', f'Y over Cross Section at $X = {x_cross_section} mm$, $Y = {y_cross_section} mm$ (mm)',
                     '$log_{10}$(mean) $W/mm^2$', True)
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(right=0.63, bottom=0.25)
    create_stat_plot('stds', f'Y over Cross Section at $X = {x_cross_section} mm$, $Y = {y_cross_section} mm$ (mm)',
                     '$log_{10}$(STD) Δ$W/mm^2$', True)


def get_pretty_photon_level(input_label):
    return f'${input_label[1]}0^{input_label[-1]}$'


def normalize_fluence_map_and_crop(f_map):
    f_map = np.log1p(f_map.astype(np.double)).squeeze()
    if len(f_map.shape) == 3:
        f_map = f_map[f_map.shape[0] // 2]
    return f_map


def plot(datasets, input_labels, output_label, plot_output_dir, dataset_name_on_rows=False, size=60, fig_size=(50, 50)):
    """
    model -> dataset name -> label -> sample
    """
    num_datasets = len(datasets) + 1
    num_samples = len(datasets['simulation'])
    os.makedirs(plot_output_dir, exist_ok=True)
    for label in input_labels:
        print(f"Generating viz for {label}")
        fig, axs = plt.subplots(num_samples, num_datasets, figsize=fig_size, subplot_kw={'xticks': [], 'yticks': []},
                                constrained_layout=True)
        # Place input label's column first
        print(f"Input Labels and Ground Truth")
        input_column_label = get_pretty_photon_level(label)
        axs[0, 0].set_title(input_column_label, {'family': 'serif', 'color': 'black', 'size': size})
        # Place output label's column last
        gt_column_label = get_pretty_photon_level(output_label)
        axs[0, -1].set_title(gt_column_label, {'family': 'serif', 'color': 'black', 'size': size})
        for j, dataset_name in enumerate(sorted(datasets['simulation'].keys())):
            if dataset_name_on_rows:
                axs[j, 0].set_ylabel(dataset_name, {'family': 'serif', 'color': 'black', 'size': size})
            input_sample = datasets['simulation'][dataset_name][label]
            gt_sample = datasets['simulation'][dataset_name][output_label]
            input_norm_sample, gt_norm_sample = normalize_fluence_map_and_crop(input_sample), normalize_fluence_map_and_crop(gt_sample)
            axs[j, 0].imshow(input_norm_sample), axs[j, -1].imshow(gt_norm_sample)
        # Do the rest of the datasets
        i = 1
        for model_name, dataset in datasets.items():
            if model_name != 'simulation':
                print(f"Dataset: {model_name}")
                axs[0, i].set_title(model_name, {'family': 'serif', 'color': 'black', 'size': size})
                for j, dataset_name in enumerate(sorted(dataset.keys())):
                    sample = dataset[dataset_name][label]
                    norm_sample = normalize_fluence_map_and_crop(sample)
                    axs[j, i].imshow(norm_sample)
                i += 1
        fig.savefig(os.path.join(plot_output_dir, f"{label}.png"))
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
