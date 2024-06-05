# from .default import _C as cfg
from hpe.config.default import _C as cfg
import yaml

def to_dict(cfg):
    return {k: v for k, v in cfg.items()}

def get_param(cfg, key):
    params = {k.split('.')[-1].lower(): v for k,v in cfg.items() if key in k}[key.lower()]
    return {k.lower(): v for k, v in yaml.safe_load(params.dump()).items()}
     

# def get_model_param(cfg):
#     params = {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'MODEL' in k}['model']
#     return yaml.safe_load(params.dump())

# def get_loss_param(cfg):
#     params = {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'LOSS' in k}['loss']
#     return yaml.safe_load(params.dump())

# def get_data_param(cfg):
#     params = {k.split('.')[-1].lower(): v for k,v in cfg.items() if 'DATA' in k}['data']
#     return yaml.safe_load(params.dump())

if __name__ == "__main__":
    import pprint
    # print(to_dict(cfg))
    print("--------------------")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(get_param(cfg, 'MODEL'))