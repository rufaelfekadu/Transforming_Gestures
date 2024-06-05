import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix



def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def setup_logger(cfg):
    import logging
    import os
    import datetime

    log_dir = os.path.join(cfg.LOG_DIR, cfg.DATA.EXP_SETUP)

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    now = datetime.datetime.now()
    log_path = os.path.join(log_dir, 'log.txt_{}'.format(now.strftime('%Y%m%d_%H%M%S')))
    
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    # fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class AverageMeter:
    def __init__(self, value=0, count=0, total=0, avg=0):
        self.count=count
        self.avg = avg
        self.total = total
        self.value = value
    
    def update(self, value, n):
        self.value = value
        self.total += value
        self.count += n
        self.avg = self.total/n

    def __str__(self):
        return f'{self.avg}'
    
def plot_sample(pred, target, label_columns):

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    fig, ax = plt.subplots(fingers.__len__(), 1, figsize=(10,10))
    fig.set_facecolor('none')
    #  sample 200 non repeting values randomly or use all
    t = min(200, pred.shape[0])
    idx_x = torch.randperm(pred.shape[0])[:t]
    #  compute average for each finger
    for i, c in enumerate(fingers):
        idx = [j for j in range(len(label_columns)) if c in label_columns[j].lower()]
        #  sample 200 non repeting values randomly
        line1 = sns.lineplot(x=idx_x, y=target[idx_x,:][:,idx].mean(dim=1), ax=ax[i], label='Leap', alpha=0.8)
        line2 = sns.lineplot(x=idx_x, y=pred[idx_x,:][:,idx].mean(dim=1), ax=ax[i], label='Predicted', alpha=0.8)
        ax[i].text(0.01, 0.5, c.capitalize(), va='center', rotation='vertical', fontsize=18, transform=ax[i].transAxes)

        # ax[i].set_title(c.capitalize(), fontsize=25, loc='center')  # Increase the font size for the title
        # remove the title
        ax[i].set_title('')
        ax[i].set_ylabel('')
        # update the font size for the ticks
        ax[i].tick_params(axis='both', which='major', labelsize=20)
        #  show legend only for the first plot
        # if i == 0:
        #     ax[i].legend()
        # else:
        #     ax[i].get_legend().remove()
        ax[i].get_legend().remove()

        # remove background color
        ax[i].set_facecolor('none')

        # Hide x-axis label for all but the last subplot
        if i != len(fingers) - 1:
            ax[i].set_xlabel('')
            ax[i].set_xticklabels([])
        else:
            ax[i].set_xlabel('Time (ms)', fontsize=20)
            ax[i].set_xticklabels([f'{i}' for i in range(0, t, 10)])

    #  set ylabel for the whole plot
    fig.text(0.001, 0.5, 'Angle (Degrees)', va='center', rotation='vertical', fontsize=20)
    plt.tight_layout()

    return fig
def plot_confussion(y_true, y_pred):
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    
    # Normalize confusion matrix
    cm = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100

    # plot confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=(8,8))
    fig.set_facecolor('none')
    ax.set_facecolor('none')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_true))
    disp.plot(cmap='Blues', ax=ax, values_format='.1f', text_kw={'fontsize':15})  # Display values as percentages
    # increase the font size of the xticks
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=15)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=15)
    # remove the x and y labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()

    return fig

def plot_emg(data):
    w,h = data.shape
    fig, ax = plt.subplots(h, 1, figsize=(10,10))
    colors = sns.color_palette("viridis", h)
    # colors = cm.viridis(np.linspace(0, 1, h))  # Create a color array

    for i in range(h):
        sns.lineplot(x=range(w), y=data[:,i], ax=ax[i], alpha=0.8, color=colors[i])
        #  remove the xticks and yticks
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_ylabel('')
        # update the yrange
        ax[i].set_ylim(-1, 1)
        # Turn off the right and top spines
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['left'].set_visible(True)
        # Only show the bottom spine for the last subplot
        if i != h - 1:
            ax[i].spines['bottom'].set_visible(False)
        else:
            ax[i].set_xlabel('Time (ms)', fontsize=20)
    plt.tight_layout()
    return fig

def plot_scatter_with_pca(data, labels, legend):

    # Sample random 1000 points
    n = min(1000, data.shape[0])
    idx = np.random.choice(data.shape[0], n, replace=False)
    data = data[idx]
    labels = labels[idx]

    # normalize
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    tsne = TSNE(n_components=2)

    # pca = PCA(n_components=2)
    reduced_data = tsne.fit_transform(data)

    # Create a color map
    # cmap = plt.cm.get_cmap('tab20', np.unique(labels).size)
    cmap = sns.color_palette("pastel", np.unique(labels).size)
    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    fig.set_facecolor('none')
    ax.set_facecolor('none')

    sns.set(style="white")

    # scatter = ax.scatter(reduced_data[:,0], reduced_data[:,1], c=labels, cmap=cmap, s=12)
    scatter = sns.scatterplot(x=reduced_data[:,0], y=reduced_data[:,1], 
                              hue=labels, palette=cmap, ax=ax, s=20)

    # remove legend
    ax.get_legend().remove()
    # Remove both ticks
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()


    return fig, None

