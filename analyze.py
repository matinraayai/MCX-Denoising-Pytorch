import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from evaluation.utils import read_mat_files, prepare_label_to_idx_mapping_for_analysis
from model.loss import SSIM, PSNR
from torch.nn import MSELoss
import torch
from config import read_analysis_cfg_file


def get_args():
    parser = argparse.ArgumentParser(description="Script for analysing filtering results for both 2D and 3D."
                                                 "it calculates both the metric results and cross section statistics.")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). '
                                                        'Refer to config.py and the config/analysis directory'
                                                        'for more info on the format.')
    return parser.parse_args()


def compute_cross_section_stats(fluence_volume, cross_section_coordinates=(50, 50), zero_nans=True, zero_infs=True):
    """
    https://3.basecamp.com/3261719/buckets/447257/todos/1194253648#__recording_1236125346
    Computes the SNR, log10 of mean and log10 of std over a cross section on the x,y-axis of the fluence volume.
    :param fluence_volume: A stack of an identical 3D simulation domain.
    :param cross_section_coordinates: A tuple specifying the location of the cross section. In case of a 2D fluence
    map, only the first element is used to select x cross section
    :param zero_nans: set the nans in the cross section to zero
    :param zero_infs: set the +/-infs in the cross section to zero
    :return: a dictionary, containing log10 of mean, log10 of std, and the SNR of the z-cross section
    """
    fluence_volume = fluence_volume.squeeze()
    if len(fluence_volume.shape) == 3:
        num_samples, x_axis_len, y_axis_len = fluence_volume.shape[0], \
                                              fluence_volume.shape[1], \
                                              fluence_volume.shape[2]
        cross_section = np.zeros((num_samples, y_axis_len), dtype=np.float64)
        x_cross_section = cross_section_coordinates[0]
        for i in range(num_samples):
            cross_section[i] = fluence_volume[i, x_cross_section, :].squeeze()

    elif len(fluence_volume.shape) == 4:
        num_samples, x_axis_len, y_axis_len, z_axis_len = fluence_volume.shape[0], \
                                                          fluence_volume.shape[1], \
                                                          fluence_volume.shape[2], \
                                                          fluence_volume.shape[3]

        cross_section = np.zeros((num_samples, z_axis_len), dtype=np.float64)
        x_cross_section, y_cross_section = cross_section_coordinates
        for i in range(num_samples):
            cross_section[i] = fluence_volume[i, x_cross_section, y_cross_section, :].squeeze()
    else:
        raise AssertionError("Fluence map either has to be a 3D tensor (for 2D simulations) or 4D tensor "
                             "(for 3D simulations)")
    means = np.log10(cross_section.mean(axis=0))
    stds = np.log10(cross_section.std(axis=0))
    if zero_nans:
        means[means != means] = 0
        stds[stds != stds] = 0
    if zero_infs:
        means[means == float('inf')] = 0
        means[means == float('-inf')] = 0
        stds[stds == float('inf')] = 0
        stds[stds == float('-inf')] = 0
    snr_results = 20 * (means - stds)
    return {'means': means, 'stds': stds, 'snr': snr_results}


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
        linestyles = ['solid', 'dotted', 'dashed', 'dashdot', ':']
        for i, (data_name, data) in enumerate(stat_dicts.items()):
            if label in data:
                if label != 'x1e9':
                    x_values = np.arange(0, len(data[label][stat_name]))
                    y_values = data[label][stat_name]
                    plt.plot(x_values, y_values, color=color, label=f"{plot_label}_{data_name}", linestyle=linestyles[i % 5])

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
    legend = f'Z over Cross Section at $X = {x_cross_section} mm$, $Y = {y_cross_section} mm$ (mm)' \
        if y_cross_section else f'Y over Cross Section at $X = {x_cross_section} mm$'
    create_stat_plot('snr', legend, 'SNR (DBs)', True)

    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(right=0.63, bottom=0.25)
    create_stat_plot('means', legend, '$log_{10}$(mean) $W/mm^2$', True)
    fig = plt.figure(figsize=(10, 6))
    fig.subplots_adjust(right=0.63, bottom=0.25)
    create_stat_plot('stds', legend, '$log_{10}$(STD) Δ$W/mm^2$', True)


def calculate_mean_snr_improvements(original_snr_array, target_snr_array):
    diff = target_snr_array - original_snr_array
    diff_nonzero = diff[0.5 < diff]
    snr_improvement = diff.mean()
    snr_improvement_non_zero = diff_nonzero.mean()
    return snr_improvement, snr_improvement_non_zero


def print_snr_improvements(datasets_stats, labels):
    print("SNR improvements")
    for label in labels:
        for data in datasets_stats:
            if data != "simulation":
                print(f"{label}: ")
                if label in datasets_stats[data]:
                    snr, non_zero_snr = calculate_mean_snr_improvements(datasets_stats["simulation"][label]["snr"],
                                                                        datasets_stats[data][label]["snr"])
                    print(f"{data} SNR: {snr}, non-zero SNR improvement: {non_zero_snr} ")


def loss_analysis(datasets, input_labels, output_label):
    input_dim = 3 if len(datasets["simulation"][output_label].shape) == 4 else 2
    loss_types = {'MSE': MSELoss(), 'SSIM': SSIM(dim=input_dim).cuda(), 'PSNR': PSNR()}

    loss_stats = {label: {data: {loss_type: 0 for loss_type in loss_types.keys()}
                          for data in datasets if data != 'simulation'}
                  for label in input_labels}

    for label in input_labels:
        for data in datasets:
            if data != "simulation":
                targets = torch.from_numpy(datasets["simulation"][output_label].squeeze()).cuda()
                preds = torch.from_numpy(datasets[data][label].squeeze()).cuda()
                for j in range(len(targets)):
                    t = torch.log1p(targets[j].unsqueeze(0).unsqueeze(0).float())
                    p = torch.log1p(preds[j].unsqueeze(0).unsqueeze(0).float())
                    for loss_type, loss_module in loss_types.items():
                        loss_stats[label][data][loss_type] += loss_module(t, p) / len(targets)
    return loss_stats


def print_loss_analysis(loss_stats):
    print("Loss Analysis\n\n\n")
    for label in loss_stats:
        print(f'Loss for label {label}:')
        for data in loss_stats[label]:
            print(f"Dataset {data}")
            for loss_type, loss in loss_stats[label][data].items():
                print(f'{loss_type}: {loss}')
    print('\n\n')


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = read_analysis_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)

    # Read in all the datasets. The datasets dictionary will have the following mapping:
    # dataset_name (e.g. simulation) -> label (e.g. 'x1e5') -> fluence map (e.g. np.ndarray(100, 100, 100))
    datasets = {}
    for dataset_name, path in cfg.dataset.paths.items():
        print(f"Reading dataset {dataset_name}:")
        datasets[dataset_name] = prepare_label_to_idx_mapping_for_analysis(read_mat_files(path))
        print("Done reading.")

    print("Calculating Stats for each dataset...")
    datasets_stats = {}
    for dataset_name, data in datasets.items():
        datasets_stats[dataset_name] = {label: compute_cross_section_stats(datasets[dataset_name][label],
                                                                           cross_section_coordinates=(cfg.cross_section.x,
                                                                                                      cfg.cross_section.y),
                                                                           zero_nans=cfg.zero_nans,
                                                                           zero_infs=cfg.zero_infs)
                                        for label in data.keys()}
    print("Done calculating. Plotting statistics...")
    plot_stats(datasets_stats, cfg.cross_section.x, cfg.cross_section.y,
               cfg.dataset.input_labels + [cfg.dataset.output_label],
               cfg.figures.fig_type, cfg.output_path)
    print("Done plotting.")

    print_snr_improvements(datasets_stats, cfg.dataset.input_labels)

    # Loss Analysis
    loss_stats = loss_analysis(datasets, cfg.dataset.input_labels, cfg.dataset.output_label)
    print_loss_analysis(loss_stats)


if __name__ == '__main__':
    main()
