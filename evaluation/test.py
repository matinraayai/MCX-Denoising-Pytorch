import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import get_model
import argparse
import tqdm
import matplotlib
import matplotlib.pyplot as plt
from config import get_cfg_defaults
from model.loss import PSNR, SSIM
import torch.nn as nn
from time import time
from evaluation.utils import visualize
from octave import octave

def get_args():
    parser = argparse.ArgumentParser(description="Model Inference")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    parser.add_argument('--checkpoint', type=str, default=None, help='path to load the checkpoint')
    parser.add_argument('--cross-section', type=int, default=None, help='Cross section to calculate the SNR plot')
    return parser.parse_args()


def ANLM_filter(x: torch.Tensor):
    x = x.cpu().numpy()
    output = torch.from_numpy(octave.mcx_filter(x, 3, 1, 2))
    # Remove the nan and inf results
    output[output != output] = 0
    output[output == float('inf')] = 0
    output[output == float('-inf')] = 0
    return output


def compute_mid_cross_section_stats(fluence_maps: torch.Tensor, x_cross_section=None):
    """
    https://3.basecamp.com/3261719/buckets/447257/todos/1194253648#__recording_1236125346
    Computes the SNR, log10 of mean and log10 of std over a cross section on the x-axis of the input.
    The input tensor contains a stack of an identical 2D simulation domain.
    :param fluence_maps: a 3D array in shape of [num_samples, x_axis, y_axis], that has stacked all the simulations
    of the same domain over the first dimension of the tensor.
    :param x_cross_section: Cross section to analyse
    :return: a dict containing the means, STDs and SNR stats over the middle of the x_axis. Each output will be a
    vector of shape [num_samples, y_axis].
    """
    num_samples = fluence_maps.shape[0]
    y_axis_len = fluence_maps.shape[2]
    cross_section = torch.zeros((num_samples, y_axis_len), dtype=torch.float32)
    if x_cross_section is None:
        x_cross_section = fluence_maps.shape[1] // 2
    for i in range(num_samples):
        cross_section[i, :] = fluence_maps[i, x_cross_section, :]
    print(cross_section)
    means = cross_section.mean(dim=0).log10()
    stds = cross_section.std(dim=0).log10()
    snr_results = 20 * (means - stds)
    return {'means': means, 'stds': stds, 'snr': snr_results}


def plot_stats(simulation_stats, prediction_states, anlm_prediction_stats,
               x_cross_section, output_dir, matplotlib_backend='Agg'):
    matplotlib.use(matplotlib_backend)
    if type(x_cross_section) == str and x_cross_section.lower() == 'middle':
        x_cross_section = 'in the middle'
    else:
        x_cross_section = f'at {x_cross_section}'

    def add_label_to_stat_plot(label, stat_name, color):
        simulation_curve = simulation_stats[label][stat_name]
        x_values = torch.arange(0, len(simulation_curve))
        plt.plot(x_values, simulation_curve, color=color, label=label)
        if label in prediction_states:
            prediction_curve = prediction_states[label][stat_name]
            plt.plot(x_values, prediction_curve, color=color, linestyle='dashed', label=label + "_pred")
            # anlm_curve = anlm_prediction_stats[label][stat_name]
            # print(anlm_curve)
            # plt.plot(x_values, anlm_curve, color=color, linestyle='dashdot', label=label + "_anlm")

    def create_stat_plot(stat_name, x_axis_label, y_axis_label):
        c_map = plt.cm.get_cmap('hsv', 10)
        i = 0
        for label in simulation_stats.keys():
            add_label_to_stat_plot(label, stat_name, c_map(i))
            i += 1
        plt.legend()
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        plt.savefig(os.path.join(output_dir, f'{stat_name}.png'))
        plt.close()

    create_stat_plot('snr', f'Y over Cross Section {x_cross_section} (mm)', 'SNR (DBs)')
    create_stat_plot('means', f'Y over Cross Section {x_cross_section} (mm)', 'Mean W/mm^2')
    create_stat_plot('stds', f'Y over Cross Section {x_cross_section} (mm)', 'STD ΔW/mm^2')


def main():
    r"""Main function."""
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = get_cfg_defaults()
    cfg.update()

    cfg.merge_from_file(args.config_file)

    cfg.freeze()
    print("Configuration details:")
    print(cfg)

    # Load model and its checkpoint
    model = get_model(**cfg.model).cuda()
    state_dict = torch.load(args.checkpoint)
    model.load_state_dict(state_dict.state_dict())

    # Different Metrics
    mse_criterion = nn.MSELoss()
    mse_loss = 0.

    ssim_criterion = SSIM(**cfg.loss.ssim).cuda()
    ssim_loss = 0.

    psnr_criterion = PSNR()
    psnr_loss = 0.

    test_dataset = OsaDataset(cfg.dataset.valid_path, cfg.dataset.input_labels, cfg.dataset.output_label,
                              False, cfg.dataset.crop_size)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=False, num_workers=cfg.dataset.dataloader_workers,
                                 pin_memory=True)

    model.train(False)

    iterator_test = tqdm.tqdm(test_dataloader)
    iterator_test.set_description(f"Inference Progress")
    test_output_dir = os.path.join(cfg.inference.output_dir)
    os.makedirs(test_output_dir, exist_ok=True)
    simulations = {label: [] for label in cfg.dataset.input_labels + [cfg.dataset.output_label]}
    predictions = {label: [] for label in cfg.dataset.input_labels}
    # anlm_predictions = {label: [] for label in cfg.dataset.input_labels}
    for i, (x_tests, y_test) in enumerate(iterator_test):
        # for label, x_test in x_tests.items():
        #     anlm_predictions[label].append(ANLM_filter(x_test))
        x_tests = {label: x_test.cuda() for label, x_test in x_tests.items()}
        y_test = y_test.cuda()
        with torch.no_grad():
            for label, x_test in x_tests.items():
                start = time()
                prediction = model(x_test)
                end = time()
                # Loss Updates
                mse_loss += mse_criterion(prediction, y_test)
                ssim_loss += ssim_criterion(prediction, y_test)
                psnr_loss += psnr_criterion(x_test, prediction)

                # visualize
                visualize(x_test.squeeze().cpu().numpy(),
                          y_test.squeeze().cpu().numpy(),
                          prediction.squeeze().cpu().numpy(),
                          os.path.join(test_output_dir, f'{i}_{label}.png'))
                iterator_test.set_postfix({"Inf. time": "{:.5f}".format(end - start)})
                # Stats array update
                simulations[label].append(torch.exp(x_test - 1))
                predictions[label].append(torch.exp(prediction - 1))
            # Append label to stats
            simulations[cfg.dataset.output_label].append(torch.exp(y_test - 1))

    simulations = {label: torch.cat(simulation).squeeze() for label, simulation in simulations.items()}
    predictions = {label: torch.cat(prediction).squeeze() for label, prediction in predictions.items()}
    # anlm_predictions = {label: torch.cat(prediction).squeeze() for label, prediction in anlm_predictions.items()}

    # torch.save(anlm_predictions, 'tmp')
    simulation_stats = {label: compute_mid_cross_section_stats(simulation) for label, simulation in simulations.items()}
    prediction_stats = {label: compute_mid_cross_section_stats(prediction) for label, prediction in predictions.items()}
    # anlm_predictions_stats = {label: compute_mid_cross_section_stats(prediction) for label, prediction in
    #                           anlm_predictions.items()}

    if args.cross_section is None:
        args.cross_section = 'Middle'
    plot_stats(simulation_stats, prediction_stats, None, args.cross_section, cfg.inference.output_dir)

    print(f"Metrics:",
          f"Mean MSE Loss: {mse_loss / len(test_dataset)}",
          f"Mean SSIM Loss: {ssim_loss / len(test_dataset)}",
          f"Mean PSNR Loss: {psnr_loss / len(test_dataset)}"
          )

    # calculate SNR improvement for each label
    # print("ANLM improvements")
    # for label in anlm_predictions_stats:
    #     if label != cfg.dataset.output_label:
    #         diff = anlm_predictions_stats[label]['snr'] - simulation_stats[label]['snr']
    #         diff_nonzero = diff[0.5 < diff]
    #         snr_improvement = diff.mean()
    #         snr_improvement_non_zero = diff_nonzero.mean()
    #         print(f"{label}: SNR improvement: {snr_improvement}, Non-zero SNR improvement: {snr_improvement_non_zero}")

    print("SNR improvements")
    for label in simulation_stats:
        if label != cfg.dataset.output_label:
            diff = prediction_stats[label]['snr'] - simulation_stats[label]['snr']
            diff_nonzero = diff[0.5 < diff]
            snr_improvement = diff.mean()
            snr_improvement_non_zero = diff_nonzero.mean()
            print(f"{label}: SNR improvement: {snr_improvement}, Non-zero SNR improvement: {snr_improvement_non_zero}")


if __name__ == '__main__':
    main()
