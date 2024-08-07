from hpe.config import cfg, dump_cfg_to_yaml
from hpe.models import EmgNet, build_model, build_optimiser
from hpe.data import build_dataloaders
from hpe.loss import build_loss
from hpe.utils.misc import set_seed, setup_logger, AverageMeter
import wandb  # Import wandb

import os
import argparse
import torch
from tabulate import tabulate
wandb.login()

def main(cfg):
    # Initialize wandb
    wandb.init(project='gesture-tracking', config=cfg)

    # setup device
    device = "cpu"

    #  setup logger
    logger = setup_logger(cfg)

    # build dataset
    dataloaders = build_dataloaders(cfg)

    # build model
    model = build_model(cfg)

    # build criterion and optimiser
    criterions = build_loss(cfg)
    optimiser, scheduler = build_optimiser(cfg, model)

    #  log the configuration
    dump_cfg_to_yaml(cfg, os.path.join(cfg.SOLVER.SAVE_DIR, 'config.yaml'))
    cfg.freeze()

    # train
    train(cfg, model, dataloaders['train'], dataloaders['val'], optimiser, scheduler, criterions, logger,wandb, device)

    # test
    test_loss = test(model, dataloaders['test'], criterions[0], device)
    logger.info(tabulate([[test_loss]], headers=['Test Loss']))

    if cfg.VIS.SAVE_TEST_SET:
        # save test_set
        save_path = os.path.join(cfg.SOLVER.SAVE_DIR, 'test_set.pth')
        torch.save(dataloaders['test'].dataset, save_path)

def train(cfg, model, train_set, val_set, optimiser, scheduler, criterions, logger,wandb, device='cpu'):

    epochs = cfg.SOLVER.NUM_EPOCHS
    best_val_loss = float('inf')
    epoch_no_improve = 0
    early_stop = False
    
    logger.info("Training started")
    #  define the headers
    headers = ['Epoch', 'Total Loss', 'TF Loss', 'Pred Loss', 'Val Loss']
    logger.info(tabulate([], headers=headers, tablefmt='plain'))
    for i in range(epochs):

        train_metrics = train_epoch(model, train_set, optimiser, scheduler, criterions, device)
        val_metrics = test(model, val_set, criterions[0])
        logger.info(tabulate([[i, *train_metrics, val_metrics]], tablefmt='plain'))
        wandb.log({'Epoch': i, 'Total Loss': train_metrics[0].avg, 'TF Loss': train_metrics[1].avg,
                   'Pred Loss': train_metrics[2].avg, 'Val Loss': val_metrics.avg})
        if val_metrics.avg < best_val_loss:
            best_val_loss = val_metrics.avg
            save_path = os.path.join(cfg.SOLVER.SAVE_DIR, 'best_model_{}.pth'.format(cfg.DATA.EXP_SETUP))
            to_save = {
                'model_state_dict': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'scheduler': scheduler.state_dict(),
                'data_setup': cfg.DATA.EXP_SETUP,
            }
            torch.save(to_save, save_path)
            
        else:
            epoch_no_improve += 1
            if epoch_no_improve > cfg.SOLVER.PATIENCE:
                logger.info("Early stopping at epoch {}".format(i))
                break

        scheduler.step(val_metrics.avg)
    wandb.finish()  # Close the wandb run
def train_epoch(model, train_loader,optimiser, scheduler, criterions, device):

    model.train()

    total_loss = AverageMeter()
    tf_loss = AverageMeter()
    pred_loss = AverageMeter()
    lamb = 0.25

    for batch in train_loader:

        input_t, _, input_f, _, _, label, gesture = batch
        input_t, input_f = input_t.to(device), input_f.to(device)
        n = input_t.shape[0]

        pred, z_t, z_f = model(input_t, input_f)

        optimiser.zero_grad()

        # compute loss
        _, l_pred = criterions[0](pred, label)
        l_tf = criterions[1](z_t, z_f)
        l_total = l_pred + lamb*l_tf

        # optimisation
        l_total.backward()
        optimiser.step()

        # update average meters
        total_loss.update(l_total, n)
        tf_loss.update(l_tf, n)
        pred_loss.update(l_pred, n)
        # Log metrics to wandb

    return total_loss, tf_loss, pred_loss

def test(model, loader, criterion, device='cpu'):

    loss = AverageMeter()
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_t, _, input_f, _, _, label, gesture = batch

            input_t, input_f = input_t.to(device), input_f.to(device)
            pred = model(input_t, input_f, return_proj=False)

            l = criterion(pred, label)

            loss.update(l[1], input_t.shape[0])

    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finger gesture tracking')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    cfg.SOLVER.SAVE_DIR = os.path.join(cfg.SOLVER.SAVE_DIR, cfg.DATA.EXP_SETUP)
    os.makedirs(cfg.SOLVER.SAVE_DIR, exist_ok=True)
    
    #  set seed
    set_seed(cfg.SEED)


    main(cfg)
