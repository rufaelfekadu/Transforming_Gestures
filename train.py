from tg.config import cfg
from tg.models import EmgNet
from tg.data import build_dataloaders
from tg.utils.misc import set_seed
import argparse


def main():

    

    pass


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.freeze()

    #  set seed
    set_seed(cfg.seed)


    main(cfg)
