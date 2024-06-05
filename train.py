from tg.config import cfg
from tg.models import EmgNet, build_model, build_optimiser
from tg.data import build_dataloaders
from tg.loss import build_loss
from tg.utils.misc import set_seed, setup_logger, AverageMeter

import argparse


def main(cfg):

    # setup device
    device = "cpu"

    #  setup logger
    logger = setup_logger(cfg.LOG_DIR)

    # build dataset
    dataloaders = build_dataloaders(cfg)

    # build model
    model = build_model(cfg)

    # build criterion and optimiser
    criterions = build_loss(cfg)
    optimiser = build_optimiser(cfg)


    # train
    train(cfg, model, dataloaders['train'], dataloaders['val'], optimiser, criterions, logger, device)

    # test
    test_loss = test(model, dataloaders['test'], criterions[0], device)
    logger.info(test_loss)

    pass

def train(cfg, model, train_loader, val_loader, optimiser, criterions, logger, device='cpu'):

    epochs = cfg.SOLVER.EPOCHS
    scheduler = None

    for i in range(epochs):
        train_metrics = train_epoch(model, train_loader, optimiser, scheduler, criterions, device)
        logger.info(train_metrics)
        val_metrics = test(model, val_loader, criterions)
        logger.info(val_metrics)

def train_epoch(model, train_loader,optimiser, scheduler, criterions, device):

    total_loss = AverageMeter()
    tf_loss = AverageMeter()
    pred_loss = AverageMeter()
    lamb = 0.25

    for batch in train_loader:

        input_t, input_f, label, gesture = batch
        input_t, input_f = input_t.to(device), input_f.to(device)
        n = input_t.shape[0]

        pred, z_t, z_f = model(input_t, input_f)

        optimiser.zero_grad()

        # compute loss
        l_pred = criterions[0](label, pred)
        l_tf = criterions[1](z_t, z_f)
        l_total = l_pred + lamb*l_tf

        # optimisation
        l_total.backwards()
        optimiser.step()
        scheduler.step(l_total)

        # update average meters
        total_loss.update(l_total, n)
        tf_loss.update(l_tf, n)
        pred_loss.update(l_pred, n)

    return total_loss, tf_loss, pred_loss

def test(model, loader, criterion, device='cpu'):

    loss = AverageMeter()

    for batch in loader:
        input_t, input_f, label, gesture = batch

        input_t, input_f = input_t.to(device), input_f.to(device)
        pred = model(input_t, input_f, return_proj=False)

        l = criterion(label, pred)

        loss.update(l, input_t.shape[0])

    return loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Finger gesture tracking')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    #  set seed
    set_seed(cfg.SEED)


    main(cfg)
