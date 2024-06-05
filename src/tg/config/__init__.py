from .default import _C as cfg

def to_dict(cfg):
    return {k: v for k, v in cfg.items()}

def get_model_param(cfg):
    return {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'MODEL' in k}

def get_loss_param(cfg):
    return {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'LOSS' in k}

def get_data_param(cfg):
    return {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'DATA' in k}

if __name__ == "__main__":
    print(to_dict(cfg))