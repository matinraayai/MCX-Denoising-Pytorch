import os
import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import Criterion, get_model
from solver.builder import build_optimizer, build_lr_scheduler
import argparse
import tqdm
from model.loss import SSIM, PSNR
import torch.nn as nn
from config import read_cfg_file
from evaluation.utils import visualize
from data.augmentation import build_train_augmentor


def get_args():
    parser = argparse.ArgumentParser(description="De-noising model training script")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    return parser.parse_args()


def main():
    # arguments
    args = get_args()

    print("Command line arguments:")
    print(args)

    # configurations
    cfg = read_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)

    model = get_model(**cfg.model).cuda()
    if cfg.model.starting_checkpoint:
        state_dict = torch.load(cfg.model.starting_checkpoint)
        model.load_state_dict(state_dict.state_dict())

    optimizer = build_optimizer(cfg, model)

    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    loss = Criterion(**cfg.loss)

    mse_criterion = nn.MSELoss()

    ssim_criterion = SSIM(**cfg.loss.ssim).cuda()

    psnr_criterion = PSNR()

    train_augmentor = build_train_augmentor(**cfg.aug)
    train_dataset = OsaDataset(cfg.dataset.train_path, cfg.dataset.input_labels,
                               cfg.dataset.output_label, True, cfg.dataset.crop_size, train_augmentor)
    valid_dataset = OsaDataset(cfg.dataset.valid_path, ['x1e5'],
                               cfg.dataset.output_label, False, cfg.dataset.crop_size)
    train_dataloader = DataLoader(train_dataset, cfg.solver.batch_size, shuffle=True,
                                  num_workers=cfg.dataset.dataloader_workers,
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False,
                                  num_workers=cfg.dataset.dataloader_workers,
                                  pin_memory=True)
    # Training
    for epoch_num in range(cfg.solver.total_iterations):
        model.train()
        iterator = tqdm.tqdm(train_dataloader)
        iterator.set_description(f"Epoch #{epoch_num}")

        for iteration, (label_batch, x_batch_train, y_batch_train) in enumerate(iterator):
            x_batch_train, y_batch_train = x_batch_train.cuda(), y_batch_train.cuda()
            optimizer.zero_grad()
            if cfg.model.noise_map:
                #TODO: Fix noise level input
                noise_level = [float(label[-1]) for label in label_batch]
                noise_map = torch.tensor([noise_level]).cuda().repeat(1, *cfg.dataset.crop_size, 1).permute(3, 0, 1, 2)
                print(noise_map.shape)
                x_batch_train = torch.cat([x_batch_train, noise_map], dim=1)
            batch_prediction = model(x_batch_train)
            loss_value = loss(y_batch_train, batch_prediction)
            loss_value.backward()
            if (iteration + 1) % cfg.solver.iteration_step == 0:
                optimizer.step()
            iterator.set_postfix({"Batch Model Loss": "{:.5f}".format(loss_value.item())})
        lr_scheduler.step()
        # Validation
        model.train(False)
        iterator_valid = tqdm.tqdm(valid_dataloader)
        iterator_valid.set_description(f"Validation Epoch #{epoch_num}")
        total_loss = 0.
        if epoch_num % cfg.solver.iteration_save == 0:
            curr_epoch_chkpt_dir = os.path.join(cfg.checkpoint_dir, str(epoch_num))
            os.makedirs(curr_epoch_chkpt_dir, exist_ok=True)
        total_mse_loss = 0.
        total_ssim_loss = 0.
        total_psnr_loss = 0.
        for i, (x_batch_valid, y_batch_valid) in enumerate(iterator_valid):
            with torch.no_grad():
                x_batch_valid, y_batch_valid = x_batch_valid['x1e5'].cuda(), y_batch_valid.cuda()
                batch_prediction = model(x_batch_valid)
                if epoch_num % cfg.solver.iteration_save == 0 and cfg.visualize:
                    visualize(x_batch_valid.squeeze().cpu(),
                              y_batch_valid.squeeze().cpu(),
                              batch_prediction.squeeze().cpu(),
                              os.path.join(curr_epoch_chkpt_dir, f'{i}.png'))
                mse_loss = mse_criterion(y_batch_valid, batch_prediction)
                psnr_loss = psnr_criterion(x_batch_valid, batch_prediction)
                ssim_loss = ssim_criterion(y_batch_valid, batch_prediction)
                loss_value = loss(y_batch_valid, batch_prediction)
                total_mse_loss += mse_loss
                total_psnr_loss += psnr_loss
                total_ssim_loss += ssim_loss
                total_loss += loss_value
                iterator_valid.set_postfix({"Model Loss": "{:.3f}".format(loss_value),
                                            "MSE Loss": "{:.3f}".format(mse_loss),
                                            "PSNR Loss": "{:.3f}".format(psnr_loss),
                                            "SSIM Loss": "{:.3f}".format(ssim_loss)})
        if epoch_num % cfg.solver.iteration_save == 0:
            torch.save(model, os.path.join(curr_epoch_chkpt_dir, 'model_chkpt.pt'))
        print(f"Model Validation Loss: {total_loss / len(valid_dataset)}\n"
              f"MSE Validation Loss: {total_mse_loss / len(valid_dataset)}\n"
              f"SSIM Validation Loss: {total_ssim_loss / len(valid_dataset)}\n"
              f"PSNR Validation Loss: {total_psnr_loss / len(valid_dataset)}")


if __name__ == '__main__':
    main()
