from .transformer import make_vvit, VViT
from .emgnet import EmgNet

from torch.optim import Adam

def build_model(cfg):
    return EmgNet(
            proj_dim=cfg.MODEL.PROJ_DIM,
            seq_length=cfg.MODEL.FRAMES,
            output_size=len(cfg.DATA.LABEL_COLUMNS)
    )

def build_optimiser(cfg):
    if cfg.SOLVER.OPTIMISER == "adam":
        return Adam()