import argparse
import os
import tqdm
from evaluation.utils import read_mat_files, prepare_label_to_idx_mapping_for_analysis
from model.loss import SSIM, PSNR
from torch.nn import MSELoss
import torch
from config import read_global_metrics_analysis_cfg_file
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description="Script for analysing global metrics of the filtering results "
                                                 "for both 2D and 3D.")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml). ')
    return parser.parse_args()


def loss_analysis(datasets, input_labels, output_label):
    input_dim = 3 if len(datasets["Simulation"][output_label].shape) == 4 else 2
    loss_types = {'MSE': MSELoss(), 'SSIM': SSIM(dim=input_dim).cuda(), 'PSNR': PSNR()}

    loss_stats = {label: {data: {loss_type: 0 for loss_type in loss_types.keys()}
                          for data in datasets if data != 'Simulation'}
                  for label in input_labels}

    for label in input_labels:
        for data in datasets:
            if data != "Simulation":
                targets = torch.from_numpy(datasets["Simulation"][output_label].squeeze()).cuda()
                # If any element of the prediction is nan, set it to zero
                preds = torch.from_numpy(datasets[data][label].squeeze()).cuda()
                iterator = tqdm.tqdm(range(len(targets)))
                if torch.isnan(preds.mean()):
                    preds[preds != preds] = 0.
                for j in iterator:
                    t = torch.log1p(targets[j].unsqueeze(0).unsqueeze(0).float())
                    p = torch.log1p(preds[j].unsqueeze(0).unsqueeze(0).float())
                    for loss_type, loss_module in loss_types.items():
                        l = (loss_module(t, p) / len(targets)).cpu().item()
                        loss_stats[label][data][loss_type] += l
                        iterator.set_postfix({"dataset": data, "label": label, "loss": l,
                                              "target mean": t.mean().item(),
                                              "prediction mean": p.mean().item()})
    return loss_stats


def save_loss_analysis(loss_stats, save_dir):
    print("Saving loss stats...")
    for label in loss_stats:
        df = pd.DataFrame(loss_stats[label])
        df.to_excel(os.path.join(save_dir, f"loss-analysis-{label}.xlsx"))
    print("Done.")


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = read_global_metrics_analysis_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)

    # Read in all the datasets. The datasets dictionary will have the following mapping:
    # dataset_name (e.g. simulation) -> label (e.g. 'x1e5') -> fluence map (e.g. np.ndarray(100, 100, 100))
    datasets = {}
    for dataset_name, path in cfg.dataset.paths.items():
        print(f"Reading dataset {dataset_name}:")
        datasets[dataset_name] = prepare_label_to_idx_mapping_for_analysis(read_mat_files(path))
        print("Done reading.")
    # Loss Analysis
    loss_stats = loss_analysis(datasets, cfg.dataset.input_labels, cfg.dataset.output_label)
    save_loss_analysis(loss_stats, cfg.output_path)


if __name__ == '__main__':
    main()
