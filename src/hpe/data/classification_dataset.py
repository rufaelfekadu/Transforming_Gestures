import torch
from torch.fft import fft, rfft
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize
from torch.utils.data import Subset

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from hpe.utils.data import read_dirs, strided_array, train_test_split_by_session
from hpe.data.transforms import JitterTransform, FrequencyTranform, RMSTransform, NormalizeTransform
from sklearn.model_selection import train_test_split

class EmgDatasetClassifier(Dataset):
    def __init__(self, cfg, data_paths, training_mode='pretrain', transforms=(None, None, None)):

        self.transform_t, self.transform_f, self.transform_c = transforms
        self.seq_len = cfg.DATA.SEGMENT_LENGTH
        self.stride = cfg.DATA.STRIDE
        # self.training_mode = training_mode
        edf_files, _ = read_dirs(data_paths)
        merged_data = []

        for i, edf_file in tqdm(enumerate(edf_files), desc='Loading data'):
            file_name = ".npz"
            np_file = os.path.splitext(edf_file)[0] + file_name
            if os.path.isfile(np_file):
                # load dataset
                loaded = np.load(np_file, allow_pickle=True)
                data = {key: loaded[key] for key in loaded.files}
                d = data['data']
                merged_data.append(d)
            else:
                # skip and remove merged data at index i
                print(f'{np_file} not found')
                continue

        if len(merged_data) == 0:
            raise FileNotFoundError(f'No data found in {cfg.DATA.PATH}')
            # return
        merged_data = np.concatenate(merged_data, axis=0)

        self.label_columns = data['label_columns']
        self.data_columns = data['data_columns']
        self.gesture_mapping = data['gesture_mapping']
        self.gesture_mapping_class = data['gesture_mapping_class']
        self.num_classes = len(self.gesture_mapping_class)

        # discritise data grouped by the last column
        # merged_data = self.discritise_data(merged_data, seq_len=self.seq_len, stride=self.stride)
        self.label = merged_data[..., len(self.data_columns):-2]
        self.gesture_class = merged_data[..., -2].astype(int)# id of the gesture
        self.gestures = merged_data[..., -1].astype(int)  # Id of the experiment

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.label[idx], self.gesture_class[idx]


        #  read exp_setups from json file


import json

exp_setups = {
    'exp0': {
        'pretrain': ['003'],
        'train': ['005/S1/P3'],
        'test': ['005/S1/P4']
    },
    'exp1': {
        'pretrain': ['004'],
        'train': ['005/S1/P2'],
        'test': ['005/S1/P2']
    },
    'exp2': {
        'pretrain': ['004'],
        'train': ['005/S1/P4'],
        'test': ['003/S1/P4'],
    },
    'exp3': {
        'pretrain': ['004'],
        'train': ['003/S1/P4'],
        'test': ['003/S1/P4'],
    },
    'exp4': {
        'pretrain': ['005'],
        'train': ['003/S1/P1', '003/S1/P2', '003/S1/P3'],
        'test': ['003/S1/P2'],
    },
    'exp5': {
        'pretrain': ['003'],
        'train': ['004/S1/P4', '004/S1/P2', '004/S1/P3'],
        'test': ['004/S1/P4'],
    },
    'exp6': {
        'pretrain': ['004'],
        'train': ['005/S1/P3', '005/S1/P2', '005/S1/P4'],
        'test': ['005/S1/P1'],
    },
    'exp7': {
        'pretrain': ['005'],
        'train': ['003', '004', '005/S1/P4', '005/S1/P2', '005/S1/P3'],
        'test': ['005/S1/P1'],
    },

}


def get_dirs_for_exp(cfg):
    data_path = cfg.DATA.PATH
    exp_path = cfg.DATA.EXP_SETUP_PATH
    exp_setups = json.load(open(exp_path, 'r'))

    if cfg.DATA.EXP_SETUP not in exp_setups.keys():
        raise ValueError(f'Invalid experiment setup {cfg.DATA.EXP_SETUP}')

    pretrain_dirs = []
    train_dirs = []
    test_dirs = []

    if 'pretrain' in exp_setups[cfg.DATA.EXP_SETUP]:
        for dir in exp_setups[cfg.DATA.EXP_SETUP]['pretrain']:
            pretrain_dirs.append(os.path.join(data_path, dir))

    if 'train' in exp_setups[cfg.DATA.EXP_SETUP]:
        for dir in exp_setups[cfg.DATA.EXP_SETUP]['train']:
            train_dirs.append(os.path.join(data_path, dir))

    if 'test' in exp_setups[cfg.DATA.EXP_SETUP]:
        for dir in exp_setups[cfg.DATA.EXP_SETUP]['test']:
            test_dirs.append(os.path.join(data_path, dir))

    return pretrain_dirs, train_dirs, test_dirs


def build_dataloaders_for_classifier(cfg, pretrain=True,rep=None):
    pretrain_dirs, train_dirs, test_dirs = get_dirs_for_exp(cfg)
    dataloaders = {}
    num_workers = cfg.SOLVER.NUM_WORKERS


    transforms = (None, None, None)


    train_set = EmgDatasetClassifier(cfg, train_dirs, training_mode='hpe',
                           transforms=transforms)
    # cfg.DATA.LABEL_COLUMNS = train_set.label_columns.tolist()
    # cfg.DATA.NUM_CLASSES = train_set.num_classes
    # cfg.MODEL.FRAMES = train_set.data.shape[1]
    # cfg.MODEL.OUTPUT_SIZE = train_set.label.shape[-1]
    unique_gestures = np.unique(train_set.gesture_mapping_class)



    if rep is None:
        rep = [np.random.randint(1, 5)]
    #  use one of the repititions as validation
    test_gestures=[]
    for k in rep:
        test_gestures += [i + f'_{k}' for i in unique_gestures]

    train_set, val_set, test_set = train_test_gesture_split_classification(train_set, test_gestures=test_gestures)
    # train_set, val_set, test_set = train_test_split_by_session(train_set)

    test_set_2 = EmgDatasetClassifier(cfg, test_dirs, training_mode='hpe')
    # _,_, test_set_2 = train_test_gesture_split(test_set_2, test_gestures=test_gestures)
    # _,_, test_set_2 = train_test_split_by_session(test_set_2)

    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                       num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                     num_workers=num_workers, persistent_workers=True, drop_last=False)
    # dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
    #                                                   num_workers=num_workers, persistent_workers=True, drop_last=False)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    # split test into validation and test

    return dataloaders,unique_gestures


def build_dataloader_classification(cfg,rep=None):
    _, train_dirs, test_dirs = get_dirs_for_exp(cfg)
    dataloaders = {}

    transforms_c = Compose([RMSTransform(),
                            NormalizeTransform(norm_type='zscore')])
    train_set = EmgDatasetClassifier(cfg, train_dirs, training_mode='classify')

    rep = np.random.randint(1, 5)

    #  use one of the repititions as validation
    unique_gestures = np.unique(train_set.gesture_mapping_class)
    if rep is None:
        rep = [np.random.randint(1, 5)]
    #  use one of the repititions as validation
    test_gestures = []
    for k in rep:
        test_gestures += [i + f'_{k}' for i in unique_gestures]

    print("gestures are:\n\t"+"\n\t".join(unique_gestures))
    print("test gestures are:\n\t"+"\n\t".join(test_gestures))
    train_set, val_set, test_set = train_test_gesture_split_classification(train_set, test_gestures=test_gestures)

    test_set_2 = EmgDatasetClassifier(cfg, test_dirs, training_mode='classify')
    _, _, test_set_2 = train_test_gesture_split_classification(test_set_2, test_gestures=test_gestures)

    num_workers = cfg.SOLVER.NUM_WORKERS

    # dataloaders['pretrain'] = None
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True,
                                                       num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
                                                     num_workers=num_workers, persistent_workers=True, drop_last=True)
    # dataloaders['test'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False,
    #                                                   num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    # split test into validation and test

    return dataloaders,unique_gestures


def train_test_gesture_split_classification(dataset, test_gestures):
    train_idx = []
    test_idx = []
    val_idx = []

    if len(dataset.gestures.shape) == 2:
        g = dataset.gestures[:, -1]
    else:
        g = dataset.gestures
    for idx, gesture in enumerate(g):
        if (gesture>=dataset.gesture_mapping.shape[0]):
            continue
        if 'rest' in dataset.gesture_mapping[gesture]:
            continue

        if dataset.gesture_mapping[gesture] in test_gestures:
            test_idx.append(idx)

        else:
            train_idx.append(idx)

    # add 20% of the test gestures to the train set
    test_idx, to_append = train_test_split(test_idx, test_size=0.2, shuffle=True, stratify=g[test_idx])
    train_idx.extend(to_append)

    train_idx, val_idx = train_test_split(train_idx, test_size=0.15, shuffle=True)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set
if __name__ == "__main__":
    from hpe.config import cfg
    import argparse
    parser = argparse.ArgumentParser(description='Finger gesture tracking decoder (Angle to pose)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # build_dataloaders(cfg)
    dataloaders = build_dataloaders_for_classifier(cfg)

    for key in dataloaders.keys():
        print(key, len(dataloaders[key]))