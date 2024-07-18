import torch
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from hpe.trainer.trainer import EmgNet
from hpe.models import EmgNet, build_model, build_optimizer
from hpe.config import cfg
from hpe.data.emgdataset import build_dataloaders
import argparse
import os
import wandb
from pytorch_lightning.loggers import WandbLogger


def setup_seed(seed):
    import random
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def main(cfg):
    # Build logger
    csv_logger = pl_loggers.CSVLogger(cfg.SOLVER.LOG_DIR, name='csv_logs')
    tb_logger = pl_loggers.TensorBoardLogger(cfg.SOLVER.LOG_DIR, name='tb_logs')
    wandb_logger = WandbLogger(project='gesture-tracking', log_model=True)
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='classification_val_loss', patience=cfg.SOLVER.PATIENCE)

    print('Finetuning model')
    try:
        # step out of the current log dir one stage back to load the pretrained model
        pretrain_path = os.path.join(os.path.dirname(cfg.SOLVER.LOG_DIR),
                                     'pretrain/checkpoints_pretrain/best_pretrained.ckpt')
        model = EmgNet.load_from_checkpoint(pretrain_path)
    except:
        model = EmgNet(cfg=cfg)

    model.stage = 'hpe'
    early_stop_callback = pl.callbacks.EarlyStopping(monitor='val_loss_c', patience=cfg.SOLVER.PATIENCE)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=os.path.join(cfg.SOLVER.LOG_DIR, 'checkpoints_finetune'), monitor='val_loss_c', save_top_k=1, mode='min')

    trainer_finetune = pl.Trainer(
        default_root_dir=os.path.join(cfg.SOLVER.LOG_DIR, 'checkpoints_finetune'),
        max_epochs=cfg.SOLVER.NUM_EPOCHS,
        logger=[tb_logger, wandb_logger],
        check_val_every_n_epoch=1,
        log_every_n_steps=len(model.train_loader),
        # limit_val_batches=0.5,
        # limit_test_batches=0.5,
        # limit_train_batches=0.05,
        callbacks=[early_stop_callback]
    )
    # finetune model
    trainer_finetune.fit(model)

    print('Testing model')
    # test model
    trainer_finetune.test(model)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description="Hand pose estimation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    # merge config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.STAGE = 'hpe'
    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.DATA.EXP_SETUP, cfg.STAGE)
    # create log dir
    os.makedirs(cfg.SOLVER.LOG_DIR, exist_ok=True)

    # setup seed
    setup_seed(cfg.SEED)

    main(cfg)