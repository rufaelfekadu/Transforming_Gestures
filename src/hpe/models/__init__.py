from .transformer import make_vvit, VViT
from .emgnet import EmgNet

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .vivit import ViViT
def build_model(cfg):
    from hpe.config import  get_param
    args = get_param(cfg, 'MODEL')
    args.update(get_param(cfg, 'TRANSFORMER'))
    if cfg.MODEL.NAME.lower() == "emgnet":
        return EmgNet(**args)
    elif cfg.MODEL.NAME.lower() == "vivit":
        return ViViT(image_size=cfg.MODEL.IMAGE_SIZE,patch_size=cfg.MODEL.PATCH_SIZE,\
                     num_classes=cfg.MODEL.OUTPUT_SIZE,num_frames=cfg.MODEL.FRAMES,\
                     dim=cfg.TRANSFORMER.D_MODEL,heads=cfg.TRANSFORMER.NUM_HEADS,\
                     depth=cfg.TRANSFORMER.NUM_LAYERS,pool=cfg.TRANSFORMER.POOL,\
                     dim_head=cfg.TRANSFORMER.DIM_HEAD,dropout=cfg.TRANSFORMER.ATT_DROPOUT,\
                     emb_dropout=cfg.MODEL.EMB_DROPOUT,scale_dim=cfg.TRANSFORMER.MLP_DIM/cfg.TRANSFORMER.D_MODEL\
                     )
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