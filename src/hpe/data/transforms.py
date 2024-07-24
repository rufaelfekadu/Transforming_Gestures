from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

import numpy as np
from torchvision import transforms
import torch


class FastICATransform:
    def __init__(self, n_components=None, random_state=0):
        self.fast_ica = FastICA(n_components=n_components, random_state=random_state)
        self.mixing_matrix = None

    def __call__(self, X):
        import copy
        if len(X.shape) == 3:
            N, S, C = X.shape
            X_ICA = copy.deepcopy(X)
            X_ICA = X_ICA.reshape(-1, C)
            X_ICA = self.fast_ica.fit_transform(X_ICA)
            self.mixing_matrix = self.fast_ica.mixing_
            return np.stack([X.reshape(N, S, C), X_ICA.reshape(N, S, C)], axis=0)
        else:
            X_ICA = copy.deepcopy(X)
            X_ICA = self.fast_ica.fit_transform(X_ICA)
            self.mixing_matrix = self.fast_ica.mixing_
            return np.stack([X, X_ICA], axis=0)

class JitterTransform:
    def __init__(self, scale=1.1):
        self.scale = scale
    def __call__(self, X):
        return X + X * np.random.normal(loc=0., scale=self.scale, size=X.shape)

class ScalingTransform:
    def __init__(self, scale=1.1):
        self.scale = scale

    def __call__(self, X):
        return X + np.random.normal(loc=0., scale=self.scale, size=X.shape)
    
class SlidingWindowTransform:
    def __init__(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def __call__(self, X):
        if len(X.shape) == 3:
            N, S, C = X.shape
            X = X.reshape(-1, C)
            X = self._sliding_window(X)
            return X.reshape(N, -1, self.window_size, C)
        else:
            return self._sliding_window(X)

    def _sliding_window(self, X):
        N, C = X.shape
        X = X.reshape(1, N, C)
        X = X.unfold(1, self.window_size, self.stride)
        X = X.permute(0, 2, 1, 3)
        return X.reshape(-1, self.window_size, C)
    
class StandardScalerTransform:
    def __init__(self):
        self.scaler = StandardScaler()

    def __call__(self, X):
        if len(X.shape) == 3:
            N, S, C = X.shape
            X = X.reshape(-1, C)
            X = self.scaler.fit_transform(X)
            return X.reshape(N, S, C)
        else:
            return self.scaler.fit_transform(X)

class RMSTransform:
    def __call__(self, X):
        #  input shape (S, C) reshape to (1,4,4)
        if sum(X.shape)==16:
            return torch.sqrt(torch.mean(X**2, axis=0)).view(1, 4, 4)
        else:
            return torch.sqrt(torch.mean(X**2, axis=1)).view(-1,1, 4, 4)

class NormalizeTransform:
    def __init__(self, norm_type='zscore'):
        self.norm_type = norm_type

    def __call__(self, X):
        if self.norm_type == 'none':  # no normalization, fast return for online performance
            return X
        is_numpy = isinstance(X, np.ndarray)

        if is_numpy:
            X = torch.tensor(X)
        axis = 0  # normalize over the channels and samples

        if self.norm_type == 'zscore':
            mean = torch.mean(X, dim=axis, keepdim=True)
            std = torch.std(X, dim=axis, keepdim=True)
            X = (X - mean) / std
        elif self.norm_type == '01':
            min_ = torch.min(X, dim=axis, keepdim=True)
            max_ = torch.max(X, dim=axis, keepdim=True)
            X = (X - min_) / (max_ - min_)
        elif self.norm_type == '-11':
            min_ = torch.min(X, dim=axis, keepdim=True)
            max_ = torch.max(X, dim=axis, keepdim=True)
            X = 2 * (X - min_) / (max_ - min_) - 1
        elif 'quantile' in self.norm_type:
            quantiles = self.norm_type.split('_')[1].split('-')
            quantiles = [float(q) for q in quantiles]
            low_quantile = torch.quantile(X, quantiles[0], dim=axis, keepdim=True)
            high_quantile = torch.quantile(X, quantiles[1], dim=axis, keepdim=True)
            X = (X - low_quantile) / (high_quantile - low_quantile)
        elif self.norm_type == 'max':
            max_ = torch.max(X, dim=axis, keepdim=True)
            X = X / max_
        else:
            raise ValueError('Invalid normalization method for EMG data')
        if is_numpy:
            X = X.numpy()
        return X
    
class ReshapeTransform:
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, X):
        return X.reshape(-1, self.shape)
    
class FilterTransform:
    def __init__(self, fs, notch_freq=50, lowcut=10, highcut=450, Q=30):
        self.fs = fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.Q = Q
        self.notch_freq = notch_freq
    
    def __call__(self, X):
        return self._filter_data(X)
    
    def _filter_data(self, X):
        
        if len(X.shape) == 3:
            N, S, C = X.shape
            X = X.reshape(-1, C)
            return self._filter_data(X).reshape(N, S, C)

        # Calculate the normalized frequency and design the notch filter
        w0 = self.notch_freq / (self.fs / 2)
        b_notch, a_notch = iirnotch(w0, self.Q)

        #calculate the normalized frequencies and design the highpass filter
        cutoff = self.lowcut / (self.fs / 2)
        sos = butter(5, cutoff, btype='highpass', output='sos')

        # apply filters using 'filtfilt' to avoid phase shift
        X = sosfiltfilt(sos, X, axis=0, padtype='even')
        X = filtfilt(b_notch, a_notch, X)

        return X
    
class FrequencyTranform:
    def __init__(self, fs, pertub_ratio=0.1):
        self.fs = fs
        self.pertub_ratio = pertub_ratio
    
    def __call__(self, X):
        aug_1 = self.remove_frequency(X)
        aug_2 = self.add_frequency(X)

        return aug_1 + aug_2
    
    def remove_frequency(self, X):
        mask = torch.FloatTensor(X.shape).uniform_() > self.pertub_ratio
        mask = mask.to(X.device)
        return X*mask
    
    def add_frequency(self, X):
        mask = torch.FloatTensor(X.shape).uniform_() > self.pertub_ratio
        mask = mask.to(X.device)
        max_amplitude = torch.max(torch.abs(X))

        random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
        petrub_matrix = mask*random_am
        return X + petrub_matrix

    
def make_transform_pretrain(cfg):
    transform_t = JitterTransform(scale=cfg.DATA.JITTER_SCALE)
    transform_f = FrequencyTranform(fs=cfg.DATA.EMG.SAMPLING_RATE, pertub_ratio=cfg.DATA.FREQ_PERTUB_RATIO)
    return transform_t, transform_f

def make_transform(cfg):
    transform = []
    if cfg.DATA.NORMALIZE:
        transform.append(StandardScalerTransform())
    if cfg.DATA.FILTER:
        transform.append(FilterTransform(fs=cfg.DATA.EMG.SAMPLING_RATE,
                                         notch_freq=cfg.DATA.EMG.NOTCH_FREQ,
                                         lowcut=cfg.DATA.EMG.LOW_FREQ,
                                         highcut=cfg.DATA.EMG.HIGH_FREQ,
                                         Q=cfg.DATA.EMG.Q))
    if cfg.DATA.ICA:
        transform.append(FastICATransform(n_components=cfg.DATA.EMG.NUM_CHANNELS))

    if len(transform) == 0:
        return None
    else:
        return transforms.Compose(transform)