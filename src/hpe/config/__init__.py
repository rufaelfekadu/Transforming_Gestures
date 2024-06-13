# from .default import _C as cfg
from hpe.config.default import _C as cfg
from yacs.config import CfgNode
import yaml

def to_dict(cfg):
    return {k: v for k, v in cfg.items()}

def get_param(cfg, key):
    params = {k.split('.')[-1].lower(): v for k,v in cfg.items() if key in k}[key.lower()]
    return {k.lower(): v for k, v in yaml.safe_load(params.dump()).items()}

def dump_cfg_to_yaml(cfg, yaml_file_path):
    with open(yaml_file_path, 'w') as f:
        f.write(cfg.dump())

def load_cfg_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    cfg = CfgNode(cfg_dict)
    return cfg

if __name__ == "__main__":
    import pprint
    # print(to_dict(cfg))
    print("--------------------")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(get_param(cfg, 'MODEL'))