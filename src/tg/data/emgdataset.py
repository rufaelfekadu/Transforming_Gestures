import torch
from torch.fft import fft, rfft
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from tg.utils.data import read_dirs, train_test_gesture_split, strided_array, train_test_split_by_session
from tg.data.transforms import JitterTransform, FrequencyTranform, RMSTransform, NormalizeTransform


class EmgDataset(Dataset):
    def __init__(self, cfg, data_paths, training_mode='pretrain', transforms=(None, None, None)):

        
        self.transform_t, self.transform_f, self.transform_c = transforms
        self.seq_len = cfg.DATA.SEGMENT_LENGTH
        self.stride = cfg.DATA.STRIDE
        self.training_mode = training_mode
        edf_files, _ = read_dirs(data_paths)
        merged_data = []

        for i, edf_file in tqdm(enumerate(edf_files), desc='Loading data'):
            file_name = ".npz"
            np_file = os.path.splitext(edf_file)[0] + file_name
            if os.path.isfile(np_file):
                # load dataset
                loaded = np.load(np_file,  allow_pickle=True)
                data = {key: loaded[key] for key in loaded.files}
                d = self.discritise_data(data['data'], seq_len=self.seq_len, stride=self.stride)
                merged_data.append(d)
            else:
                # skip and remove merged data at index i
                print(f'{np_file} not found')
                continue

        if len(merged_data) == 0:
            print(f'No data found in {cfg.DATA.PATH}')
            return
        merged_data = np.concatenate(merged_data, axis=0)
        
        self.label_columns = data['label_columns']
        self.data_columns = data['data_columns']
        self.gesture_mapping = data['gesture_mapping']
        self.gesture_mapping_class = data['gesture_mapping_class']
        self.num_classes = len(self.gesture_mapping_class)

        # discritise data grouped by the last column
        # merged_data = self.discritise_data(merged_data, seq_len=self.seq_len, stride=self.stride)

        self.data = merged_data[:, :, 0:len(self.data_columns)]
        self.label = merged_data[:, :, len(self.data_columns):-2]
        self.gesture_class = merged_data[:, :, -2]
        self.gestures = merged_data[:, :, -1]


        #  to tensor
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.gesture_class = torch.tensor(self.gesture_class, dtype=torch.long)
        self.gestures = torch.tensor(self.gestures, dtype=torch.long)

        #  fft 
        self.data_f = fft(self.data, dim=1).abs()
        self.data_c = self.transform_c(self.data).float()
        
        if self.training_mode == 'pretrain':
            # time augmentations apply jitter augmentation
            self.aug1_t = self.transform_t(self.data).float()
            # frequency augmentations 
            self.aug1_f = self.transform_f(self.data_f).float()
        
        elif self.training_mode == 'classify':
            self.data = self.transform_c(self.data).float()

    def discritise_without_grouping(self, data, seq_len=150, stride=5):
        # Initialize an empty list to store the strided arrays
        strided_arrays = []

        # Iterate over the groups
        # Convert the group to a numpy array
        # Generate the strided array and append it to the list
        # assert the shape of the array is greater than the sequence length
        if data.shape[0] > seq_len:
            strided_arrays.append(strided_array(data, seq_len, stride))
        else:
            print(f'Skipping, not enough data')

        # Concatenate the strided arrays into a single array and return it
        return np.concatenate(strided_arrays, axis=0)
    
    def discritise_data(self, data, seq_len=150, stride=5):
        data = pd.DataFrame(data)
        grouped = data.groupby(data.columns[-1], sort=False)  # Update: Disable sorting by column

        # Initialize an empty list to store the strided arrays
        strided_arrays = []

        # Iterate over the groups
        for _, group in grouped:
            # Convert the group to a numpy array
            array = np.array(group)
            # Generate the strided array and append it to the list
            # assert the shape of the array is greater than the sequence length
            if array.shape[0] > seq_len:
                strided_arrays.append(strided_array(array, seq_len, stride))
            else:
                print(f'Skipping {group.iloc[0][data.columns[-1]]}, not enough data')

        # Concatenate the strided arrays into a single array and return it
        return np.concatenate(strided_arrays, axis=0)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.training_mode == 'pretrain':
            return self.data[idx], self.aug1_t[idx], self.data_f[idx], \
                self.aug1_f[idx], self.label[idx], self.gesture_class[idx], 
    
        elif self.training_mode == 'classify':
            return self.data[idx], 0 , 0, \
                0, 0, self.gesture_class[idx]
        else:
            return self.data[idx], 0, self.data_f[idx], \
                0, self.data_c[idx], self.label[idx], self.gesture_class[idx] 


#  read exp_setups from json file
import json


exp_setups = {
    'exp0':{
        'pretrain': ['003'],
        'train': ['005/S1/P3'],
        'test': ['005/S1/P4']
    },
    'exp1':{
        'pretrain': ['004'],
        'train': ['005/S1/P2'],
        'test': ['005/S1/P2']
    },
    'exp2':{
        'pretrain': ['004'],
        'train': ['005/S1/P4'],
        'test': ['003/S1/P4'],
    },
    'exp3':{
        'pretrain': ['004'],
        'train': ['003/S1/P4'],
        'test': ['003/S1/P4'],
    },
    'exp4':{
        'pretrain': ['005'],
        'train': ['003/S1/P1', '003/S1/P2', '003/S1/P3'],
        'test': ['003/S1/P2'],
    },
    'exp5':{
        'pretrain': ['003'],
        'train': ['004/S1/P4', '004/S1/P2', '004/S1/P3'],
        'test': ['004/S1/P4'],
    },
    'exp6':{
        'pretrain': ['004'],
        'train': ['005/S1/P3', '005/S1/P2', '005/S1/P4'],
        'test': ['005/S1/P1'],
    },
    'exp7':{
        'pretrain': ['005'],
        'train': ['003','004','005/S1/P4', '005/S1/P2', '005/S1/P3'],
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

def build_dataloaders(cfg, pretrain=True):

    pretrain_dirs, train_dirs, test_dirs = get_dirs_for_exp(cfg)
    dataloaders = {}
    num_workers = cfg.SOLVER.NUM_WORKERS

    # transforms
    transforms_c = Compose([RMSTransform(),
                            NormalizeTransform(norm_type='zscore')])
    transforms_t = Compose([JitterTransform(scale=cfg.DATA.JITTER_SCALE)])
    transforms_f = Compose([FrequencyTranform(fs=cfg.DATA.EMG.SAMPLING_RATE, pertub_ratio=cfg.DATA.FREQ_PERTUB_RATIO)])


    data_paths = pretrain_dirs
    try:
        if pretrain:
            pretrain_set = EmgDataset(cfg, data_paths, training_mode='pretrain',
                                transforms = (transforms_t, transforms_f, transforms_c))
            dataloaders['pretrain'] = torch.utils.data.DataLoader(pretrain_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=num_workers, persistent_workers=True)

    except:
        print('No pretrain data found')
        dataloaders['pretrain'] = None
    train_set = EmgDataset(cfg, train_dirs, training_mode='hpe', transforms=(transforms_t, transforms_f, transforms_c))
    cfg.DATA.LABEL_COLUMNS = train_set.label_columns.tolist()
    cfg.DATA.NUM_CLASSES = train_set.num_classes

    rep = np.random.randint(1,5)

    #  use one of the repititions as validation
    unique_gestures = np.unique(train_set.gesture_mapping_class)
    test_gestures = [i+f'_{rep}' for i in unique_gestures]

    train_set, val_set, test_set = train_test_gesture_split(train_set, test_gestures=test_gestures)
    # train_set, val_set, test_set = train_test_split_by_session(train_set)

    # test_set_2 = EmgDataset(cfg, test_dirs, training_mode='hpe', transforms=(transforms_t, transforms_f, transforms_c))
    # _,_, test_set_2 = train_test_gesture_split(test_set_2, test_gestures=test_gestures)
    # _,_, test_set_2 = train_test_split_by_session(test_set_2)


    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    # dataloaders['test_2'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    # split test into validation and test

    return dataloaders


def build_dataloader_classification(cfg):
    _, train_dirs, test_dirs = get_dirs_for_exp(cfg)
    dataloaders = {}

    transforms_c = Compose([RMSTransform(),
                            NormalizeTransform(norm_type='zscore')])
    train_set = EmgDataset(cfg, train_dirs, training_mode='classify', transform_c=transforms_c)

    rep = np.random.randint(1,5)

    #  use one of the repititions as validation
    unique_gestures = np.unique(train_set.gesture_mapping_class)
    test_gestures = [i+f'_{rep}' for i in unique_gestures]

    train_set, val_set, test_set = train_test_gesture_split(train_set, test_gestures=test_gestures)

    transforms_c = Compose([RMSTransform(),
                            NormalizeTransform(norm_type='zscore')])
    
    test_set_2 = EmgDataset(cfg, test_dirs, training_mode='classify', transform_c=transforms_c)
    _,_, test_set_2 = train_test_gesture_split(test_set_2, test_gestures=test_gestures)

    num_workers = cfg.SOLVER.NUM_WORKERS

    dataloaders['pretrain'] = None
    dataloaders['train'] = torch.utils.data.DataLoader(train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['test'] = torch.utils.data.DataLoader(test_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    dataloaders['test_2'] = torch.utils.data.DataLoader(test_set_2, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=num_workers, persistent_workers=True, drop_last=True)
    # split test into validation and test

    return dataloaders

if __name__ == "__main__":
    from tg.config import cfg

    # build_dataloaders(cfg)
    dataloaders = build_dataloaders(cfg)
    print(dataloaders.keys())