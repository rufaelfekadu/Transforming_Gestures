from .transformer import make_vvit, VViT
from .emgnet import EmgNet
from .emgnet_new_vivit import make_emgnet_new
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .vivit import ViViT
from .vivit import make_vivit as make_vivit_new
def build_model(cfg):
    from hpe.config import  get_param
    args = get_param(cfg, 'MODEL')
    args.update(get_param(cfg, 'TRANSFORMER'))
    if cfg.MODEL.NAME.lower() == "emgnet":
        return EmgNet(**args)
    elif cfg.MODEL.NAME.lower() == "vivit":
        return make_vivit_new(cfg)
    if cfg.MODEL.NAME.lower() == "emgnet_new":
        return make_emgnet_new(cfg)
    else:
        raise AttributeError(f"Cannot find model named: {cfg.MODEL.NAME}")
def build_optimizer(cfg, model):
    if cfg.SOLVER.OPTIMIZER == "adam":
        learning_rate = cfg.SOLVER.LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        optimiser = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        scheduler=None
        if cfg.SOLVER.REDUCE_LR_FACTOR<1:
            # Add ReduceLROnPlateau scheduler
            scheduler = ReduceLROnPlateau(optimiser, 'min', patience=cfg.SOLVER.REDUCE_LR_PATIENCE, factor=cfg.SOLVER.REDUCE_LR_FACTOR,threshold=1e-3,cooldown=5)

        return optimiser, scheduler