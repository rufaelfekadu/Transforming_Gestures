from hpe.config import load_cfg_from_yaml
from hpe.visualize.hand import Hands
import argparse


def main(cfg):

    hands = Hands(cfg)
    hands.run_from_pretrained()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finger gesture tracking')
    parser.add_argument('--config', type=str, default='output/exp1/config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg = load_cfg_from_yaml(args.config)
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    main(cfg)