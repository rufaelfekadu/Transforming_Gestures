from .contrastive import NTXentLoss_poly
from .neuroloss import NeuroLoss

def build_loss(cfg):
    return NeuroLoss(metric=cfg.LOSS.METRIC), \
           NTXentLoss_poly(device=cfg.DEVICE,
                           batch_size=cfg.SOLVER.BATCH_SIZE,
                           temperature=cfg.LOSS.TEMPRATURE,
                           use_cosine_similarity=cfg.LOSS.USE_COSINE)
