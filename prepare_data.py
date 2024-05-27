import numpy as np
import pandas as pd
import os
import argparse
import glob
from tqdm import tqdm

from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt, stft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import FastICA
from sklearn.preprocessing import LabelEncoder

from tg.utils.data import read_emg_v1, read_leap, build_emg_columns, build_leap_columns, strided_array
from tg.config import cfg

def read_dirs(data_path):

    if isinstance(data_path, str):
        data_path = [data_path]
    all_files = []
    for path in data_path:
        if not os.path.isdir(path):
            raise ValueError(f'{path} is not a directory')
        else:
            print(f'Reading data from {path}')
            all_files += [f for f in glob.glob(os.path.join(path, '**/*'), recursive=True) if os.path.splitext(f)[1] in ['.edf', '.csv']]
    
    edf_files = sorted([file for file in all_files if file.endswith('.edf')])
    csv_files = sorted([file for file in all_files if file.endswith('.csv')])

    return edf_files, csv_files

def merge_data(emg_data, leap_data):
    #  ajust the time
    start_time = max(min(emg_data.index), min(leap_data.index))
    end_time = min(max(emg_data.index), max(leap_data.index))

    emg_data = emg_data[start_time:end_time]
    leap_data = leap_data[start_time:end_time]

    data = pd.merge_asof(emg_data, leap_data, left_index=True, right_index=False, right_on='time', direction='backward', tolerance=pd.to_timedelta(10, unit='ms'))
    data['gesture_class'] = data['gesture'].apply(lambda x: x.split('_')[0])
    
    # data['time_diff'] = (data.index - data['time_leap']).dt.total_seconds()
    # data.drop(columns=['timestamp', 'frame_id', 'time_leap'], inplace=True)


    #  reorder columns to have gesture at the end
    if 'gesture' in data.columns:
        data = data[[col for col in data.columns if col != 'gesture'] + ['gesture']]

    return data

def _filter_data(data: np.ndarray, notch_freq=50, fs=250, Q=30, low_freq=30, ) -> np.ndarray:

        # Calculate the normalized frequency and design the notch filter
        w0 = notch_freq / (fs / 2)
        b_notch, a_notch = iirnotch(w0, Q)

        #calculate the normalized frequencies and design the highpass filter
        cutoff = low_freq / (fs / 2)
        sos = butter(4, cutoff, btype='highpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        data = sosfiltfilt(sos, data, axis=0, padtype='even')
        data = filtfilt(b_notch, a_notch, data, padtype='even')



        return data

def discritise_data(data, seq_len=150, stride=5):
        # assert gesture is in the last column
        assert data.columns[-1] == 'gesture', 'gesture should be the last column'
        grouped = data.groupby('gesture')

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
                print(f'Skipping {group.iloc[0]["gesture"]}, not enough data')

        # Concatenate the strided arrays into a single array and return it
        return np.concatenate(strided_arrays, axis=0)

def _prepare_data( data_path, label_path, index=0, lock=None, event=None):

    results = {}
    data =  read_emg_v1(data_path)
    label = read_leap(label_path, rotations=True, positions=False, visualisation=False)

    data_columns = build_emg_columns()
    label_columns = build_leap_columns(full=False)

    results['data_columns'] = data_columns
    results['label_columns'] = label_columns

    # normalise and filter the data
    data[data_columns] = StandardScaler().fit_transform(data[data_columns])
    data[data_columns] = _filter_data(data[data_columns], fs=250)
    data[data_columns] = StandardScaler().fit_transform(data[data_columns])

    #  merge the data
    data = merge_data(data, label)

    # label encoder
    le = LabelEncoder()
    data['gesture'] = le.fit_transform(data['gesture'])
    results['gesture_mapping'] = le.classes_

    data['gesture_class'] = le.fit_transform(data['gesture_class'])
    results['gesture_mapping_class'] = le.classes_

    # interpolate missing values
    data = data.groupby('gesture').filter(lambda x: x.isnull().sum()[label_columns[0]] < 300)
    for column in label_columns:
        data[column] = data.groupby('gesture')[column].transform(lambda x: x.fillna(x.mean()))
    #  descritise the data
    # data = discritise_data(data)

    results['data'] = data.values

    return results


def prepare_data(cfg):

    seq_len = cfg.DATA.SEGMENT_LENGTH
    stride = cfg.DATA.STRIDE

    for edf_file, csv_file in tqdm(zip(*read_dirs(cfg.DATA.PATH))):
        file_name = ".npz"
        np_file = os.path.splitext(edf_file)[0] + file_name
        if os.path.isfile(np_file):
            print(f'{np_file} already exists')
            continue
        else:
            data = _prepare_data(edf_file, csv_file)
            np.savez(np_file, **data)
            print(f'Saved {np_file}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hand pose estimation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    # merge config file
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)


    prepare_data(cfg)