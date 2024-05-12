from .default import _C as cfg

def to_dict(cfg):
    return {k: v for k, v in cfg.items()}