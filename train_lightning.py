import argparse
import os
from datetime import datetime
from hpe.config import cfg, dump_cfg_to_yaml
from hpe.models import build_model
from hpe.data import build_dataloaders
from hpe.loss import build_loss
from hpe.utils.misc import set_seed, setup_logger
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

wandb.login()
wandb.require("core")

class EmgNetLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super(EmgNetLightningModule, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.criterions = build_loss(cfg)
        self.optimiser, self.scheduler = build_optimizer(cfg, self.model)
        self.save_hyperparameters()

    def forward(self, input_t, input_f):
        return self.model(input_t, input_f)

    def configure_optimizers(self):
        if self.scheduler:
            return [self.optimiser], [self.scheduler]
        return self.optimiser

    def training_step(self, batch, batch_idx):
        input_t, _, input_f, _, _, label, gesture = batch
        input_t, input_f = input_t.to(self.device), input_f.to(self.device)
        pred, z_t, z_f = self(input_t, input_f)
        label = label.to(self.device)
        _, l_pred = self.criterions[0](pred, label)
        l_tf = self.criterions[1](z_t, z_f)
        l_total = l_pred + self.cfg.LOSS.LAMBDA * l_tf

        self.log('train_loss', l_total)
        return l_total

    def validation_step(self, batch, batch_idx):
        input_t, _, input_f, _, _, label, gesture = batch
        input_t, input_f = input_t.to(self.device), input_f.to(self.device)
        pred, _, _ = self(input_t, input_f)
        label = label.to(self.device)
        _, l_pred = self.criterions[0](pred, label)
        self.log('val_loss', l_pred)

    def test_step(self, batch, batch_idx):
        input_t, _, input_f, _, _, label, gesture = batch
        input_t, input_f, label = input_t.to(self.device), input_f.to(self.device), label.to(self.device)
        pred = self(input_t, input_f)
        l = self.criterions[0](pred, label)
        self.log('test_loss', l[1])

def main(cfg):
    set_seed(cfg.SEED)
    wandb.init(project='gesture-tracking', config=cfg)
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    cfg.DEVICE = device

    logger = setup_logger(cfg)
    dataloaders = build_dataloaders(cfg)

    model = EmgNetLightningModule(cfg)
    trainer = pl.Trainer(max_epochs=cfg.SOLVER.NUM_EPOCHS, gpus=1 if device == 'cuda' else 0)

    trainer.fit(model, DataLoader(dataloaders['train']), DataLoader(dataloaders['val']))
    trainer.test(model, DataLoader(dataloaders['test']))

    dump_cfg_to_yaml(cfg, os.path.join(cfg.SOLVER.SAVE_DIR, 'config.yaml'))
    cfg.freeze()
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finger gesture tracking')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    parser.add_argument('--cluster', nargs=1, default=False, help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.SOLVER.SAVE_DIR = os.path.join(cfg.SOLVER.SAVE_DIR, cfg.DATA.EXP_SETUP)

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    cfg.STARTING_TIME_STEMP = formatted_time
    os.makedirs(cfg.SOLVER.SAVE_DIR, exist_ok=True)
    main(cfg)
