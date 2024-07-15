from .transformer import make_vvit, VViT
from .emgnet import EmgNet

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def build_model(cfg):
    from hpe.config import  get_param
    args = get_param(cfg, 'MODEL')
    args.update(get_param(cfg, 'TRANSFORMER'))
    return EmgNet(**args)


def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == "adam":
        learning_rate = cfg.SOLVER.LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Add ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(optimiser, 'min', patience=3, factor=0.5, verbose=True)

        return optimiser, scheduler