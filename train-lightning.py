import torch
from torch.utils.data import DataLoader
from data.dataset import OsaDataset
from model.builder import Criterion, get_model
from solver.builder import build_optimizer, build_lr_scheduler
import argparse
from model.loss import SSIM, PSNR
import torch.nn as nn
from config import read_training_cfg_file
from data.augmentation import build_train_augmentor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def get_args():
    parser = argparse.ArgumentParser(description="Denoising model training script")
    parser.add_argument('--config-file', type=str, help='configuration file (yaml)')
    return parser.parse_args()


class TrainLightningModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = get_model(**cfg.model)
        self.loss = Criterion(**cfg.loss).to(self.device)
        if cfg.model.starting_checkpoint:
            state_dict = torch.load(cfg.model.starting_checkpoint)
            self.model.load_state_dict(state_dict.state_dict())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        # if cfg.model.noise_map:
        #     #TODO: Fix noise level input
        #     noise_level = [float(label[-1]) for label in label_batch]
        #     noise_map = torch.tensor([noise_level]).cuda().repeat(1, *cfg.dataset.crop_size, 1).permute(3, 0, 1, 2)
        #     print(noise_map.shape)
        #     x_batch_train = torch.cat([x_batch_train, noise_map], dim=1)
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('batch_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return self.loss(y_hat, y)

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        mse_criterion = nn.MSELoss()
        ssim_criterion = SSIM(**self.cfg.loss.ssim).to(self.device)
        psnr_criterion = PSNR()
        y_hat = self.model(x)
        mse_loss = mse_criterion(y, y_hat)
        ssim_loss = ssim_criterion(y, y_hat)
        psnr_loss = psnr_criterion(y, y_hat)
        self.log_dict({'MSE': mse_loss, 'SSIM': ssim_loss, 'PSNR': psnr_loss})

    def configure_optimizers(self):
        optimizer = build_optimizer(self.cfg, self.model)
        lr_scheduler = build_lr_scheduler(self.cfg, optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        train_augmentor = build_train_augmentor(**self.cfg.aug)
        train_dataset = OsaDataset(self.cfg.dataset.train_path, self.cfg.dataset.input_labels,
                                   self.cfg.dataset.output_label, True, self.cfg.dataset.crop_size, train_augmentor)
        train_dataloader = DataLoader(train_dataset, self.cfg.solver.batch_size, shuffle=True,
                                      num_workers=self.cfg.dataset.dataloader_workers,
                                      pin_memory=True)
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = OsaDataset(self.cfg.dataset.valid_path, ['x1e5'],
                                   self.cfg.dataset.output_label, True, self.cfg.dataset.crop_size)
        valid_dataloader = DataLoader(valid_dataset, 1, shuffle=False,
                                      num_workers=self.cfg.dataset.dataloader_workers,
                                      pin_memory=True)
        return valid_dataloader


def main():
    # arguments
    args = get_args()
    print("Command line arguments:")
    print(args)
    # configurations
    cfg = read_training_cfg_file(args.config_file)
    print("Configuration details:")
    print(cfg)
    module = TrainLightningModule(cfg)
    if not os.path.exists(cfg.checkpoint_dir):
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    trainer = Trainer(default_root_dir=".",
                      gpus=-1, num_nodes=1, max_epochs=cfg.solver.total_iterations, accelerator='ddp',
                      plugins=DDPPlugin(find_unused_parameters=False),
                      logger=TensorBoardLogger(save_dir=cfg.checkpoint_dir, name='experiment'),
                      callbacks=[ModelCheckpoint(monitor='SSIM')])
    trainer.fit(module)


if __name__ == '__main__':
    main()
