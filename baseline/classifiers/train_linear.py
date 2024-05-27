
from hpe.data.emgdataset import build_dataloader_classification, build_dataloaders
from hpe.model.linear import make_linear_classifiers
from hpe.model.clustering import make_clustering
from hpe.model.rc import RC
from hpe.config import cfg
from hpe.utils.misc import plot_scatter_with_pca, plot_confussion
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
import matplotlib.patches as patches
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def compute_topk(outputs, labels, k):
    topk_predictions = np.argsort(outputs, axis=1)[:, -k:]
    correct = np.equal(topk_predictions, labels.reshape(-1, 1))
    return correct.sum()

def get_data(loader, c=True):
    x, y = [], []
    for data in loader:
        if c:
            x.append(data[4].numpy())
        else:
            x.append(data[0].numpy())
        y.append(data[-1][:, 0].numpy())
    return np.concatenate(x, axis=0), np.concatenate(y, axis=0)

def set_seed(seed):
    from reservoirpy import set_seed, verbosity
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    verbosity(0)

def main(cfg):

    # build data loaders
    dataloaders = build_dataloaders(cfg, pretrain=False)
    classes = dataloaders['train'].dataset.dataset.gesture_mapping_class
    num_classes = len(classes)

    train_x, train_y = get_data(dataloaders['train'])
    test_x, test_y = get_data(dataloaders['test'])

    train_x = train_x.reshape(train_x.shape[0], -1)
    test_x = test_x.reshape(test_x.shape[0], -1)

    # print data stat
    print(f"Train data: {train_x.shape[0]}")
    print(f"Test data: {test_x.shape[0]}")

    linear_classifiers = make_linear_classifiers(cfg)
    clustering = make_clustering(cfg)

    # build tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(cfg.SOLVER.LOG_DIR, 'linear_classifier/tb_logs'))

    # plot a scatter plot of the data
    fig = plot_scatter_with_pca(train_x, train_y)
    writer.add_figure('scatter plot after RMS', fig)

    for name, classifier in linear_classifiers.items():
        print(f"Training {name}")
        #  fit model
        classifier.fit(train_x, train_y)
        #  evaluate model
        acc = classifier.score(test_x, test_y)
        print(cfg.DATA.EXP_SETUP+','+'RMS_'+name +','+str(acc), file=open('outputs/linear_results.csv', 'a'),)
        writer.add_scalar(f'linear_classifier_acc/{name}', acc)

        pred = classifier.predict(test_x)

        # compute topk
        for k in [1, 3, 5]:
            pred_proba = classifier.predict_proba(test_x)
            correct = compute_topk(pred_proba, test_y, k)
            print(f"Top-{k} accuracy: {correct / len(test_y)}")
            writer.add_scalar(f'linear_classifier_top_{k}_acc/{name}', correct / len(test_y))

        # print the value counts for test_x
        # disp = ConfusionMatrixDisplay.from_predictions(test_y, pred, labels= np.unique(test_y), display_labels=classes[np.unique(test_y)])
        # disp.plot(xticks_rotation='vertical')
        fig = plot_confussion(test_y, pred)
        fig.savefig('outputs/confusion/'+cfg.DATA.EXP_SETUP+'_rms_'+name+'.png')
        writer.add_figure(f'linear_classifier_confusion_matrix/{name}', fig)

    for name, cluster in clustering.items():
        print(f"Training {name}")

        # pca = PCA(n_components=2)
        x = np.concatenate((train_x, test_x), axis=0)
        # x = pca.fit_transform(x)
        # print number of null values
        print(f"Number of null values: {np.sum(np.isnan(x))}")
        clusters = cluster.fit_predict(x)

        test_clusters = clusters[len(train_x):]

        labels = np.zeros_like(test_clusters)
        for i in range(num_classes):
            mask = (clusters[:len(train_x)] == i)
            labels[test_clusters == i] = mode(train_y[mask], keepdims=True)[0]

        # Assign each cluster to the most common class in the cluster
        # labels = np.zeros_like(test_clusters)
        # for i in range(num_classes):
        #     mask = (test_clusters == i)
        #     labels[mask] = mode(test_y[mask], keepdims=True)[0]

        # Compute accuracy
        accuracy = accuracy_score(test_y, labels)
        print("Accuracy: ", accuracy)
        writer.add_scalar(f'linear_classifier_acc/{name}', accuracy)

        # disp = ConfusionMatrixDisplay.from_predictions(test_y, labels, labels= np.unique(test_y), display_labels=classes[np.unique(test_y)])
        # disp.plot(xticks_rotation='vertical')
        # plt.tight_layout()

        fig = plot_confussion(test_y, pred)
        fig.savefig('outputs/confusion/'+cfg.DATA.EXP_SETUP+'_rms_'+name+'.png')
        writer.add_figure(f'linear_classifier_confusion_matrix/{name}', fig)

        # cmap = plt.cm.get_cmap('hsv', num_classes)
        # Plot scatter plot with colors reflecting the original labels
        # plt.scatter(x[len(train_x):, 0], x[len(train_x):, 1], c=test_y, cmap=cmap)

        # centers = cluster.cluster_centers_
        # plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
        # legend_elements = [Line2D([0], [0], marker='o', color=cmap(i), label=classes[i],
        #                         lw=4, markersize=10) for i in range(len(classes))]
        # plt.legend(handles=legend_elements, loc='best')

        # Plot a boundary around each cluster
        # ax = plt.gca()
        # for i in range(num_classes):
        #     cluster_points = x[cluster.labels_ == i]
        #     radius = max(np.sum((cluster_points - centers[i])**2, axis=1)**0.5)
        #     circle = patches.Circle((centers[i, 0], centers[i, 1]), radius, fill=False)
        #     ax.add_patch(circle)

        # plt.show()
        # plt.title(name)
        # plt.show()
        # writer.add_figure(f'linear_classifier/{name}_scatter_plot', plt.gcf())
        # plt.close()

    # reservoir network
    print("Training Reservoir Computer")
    train_x, train_y = get_data(dataloaders['train'], c=False)
    test_x, test_y = get_data(dataloaders['test'], c=False)
    classes = dataloaders['train'].dataset.dataset.gesture_mapping_class
    num_classes = len(classes)

    rc = RC(units=500, lr=0.1, sr=0.9, ridge=1e-7, output_dim=num_classes, verbose=True)
    rc.fit(train_x, train_y)
    acc = rc.score(test_x, test_y)
    print(f"Accuracy: {acc}")
    writer.add_scalar(f'linear_classifier_acc/RC', acc)

    pred = rc.predict(test_x)
    # compute topk
    for k in [1, 3, 5]:
        correct = compute_topk(pred.squeeze(1), test_y, k)
        print(f"Top-{k} accuracy: {correct / len(test_y)}")
        writer.add_scalar(f'linear_classifier_top_{k}_acc/RC', correct / len(test_y))

    pred = np.argmax(pred, axis=2)
    disp = ConfusionMatrixDisplay.from_predictions(test_y, pred, labels= np.unique(test_y), display_labels=classes[np.unique(test_y)])
    disp.plot(xticks_rotation='vertical')
    plt.tight_layout()
    writer.add_figure(f'linear_classifier_confusion_matrix/RC', plt.gcf())
    writer.close()



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Example')
    parser.add_argument('--config', default='config.yaml', type=str, help='path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')

    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.DATA.EXP_SETUP)

    # set seed
    set_seed(cfg.SEED)

    main(cfg)