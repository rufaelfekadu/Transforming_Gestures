from hpe.config import cfg
from hpe.data.emgdataset import build_dataloaders
from hpe.loss.neuroloss import NeuroLoss
from hpe.model.linear import make_linear_classifiers
from hpe.model.clustering import make_clustering
from hpe.utils.misc import setup_seed

import torch
import numpy as np
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from scipy.stats import mode
import argparse
import os
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from model import NeuroPose

def compute_topk(outputs, labels, k):
    _, topk_predictions = outputs.topk(k, dim=1)
    correct = topk_predictions.eq(labels.view(-1, 1).expand_as(topk_predictions))
    return correct.sum().item()

def train(train_loader, model, optimizer, criterion, epoch, writer, device):
    model.train()
    running_loss = 0.0
    total = 0
    epoch_loop = tqdm(train_loader, desc=f"Epoch {epoch}", leave=bool(epoch==len(train_loader)-1))
    for i, data in enumerate(epoch_loop, 0):
        inputs, _, _, _, _, labels, _ = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss[1].backward()
        optimizer.step()
        running_loss += loss[1].item()
        total += labels.size(0)
        epoch_loop.set_postfix({'loss': running_loss / total})
    epoch_loop.reset()
    writer.add_scalar('neuropose_training_loss', running_loss / len(train_loader), epoch)

def validate(test_loader, model, criterion, epoch, writer, device, stage='val'):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, _, _, _, _, labels, _ = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss[1].item()
    writer.add_scalar(f'neuropose_{stage}_loss', running_loss / len(test_loader), epoch)


def get_features(loader, model, device):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, _, _, _, _, _, label = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            features.append(outputs[:,-1,:].cpu().numpy())
            labels.append(label[:,-1].cpu().numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)

def score_linear(class_name, pred,labels,classes, summary_writer):
    acc = accuracy_score(labels, pred)
    summary_writer.add_scalar(f'classification after neuropose using {class_name}', acc)
    print(cfg.DATA.EXP_SETUP+','+'Neuropose_'+class_name +','+str(acc), file=open('outputs/linear_results.csv', 'a'),)

    # plot confusion matrix
    cm = ConfusionMatrixDisplay.from_predictions(labels, pred, labels=np.unique(labels), display_labels=classes[np.unique(labels)])
    cm.plot(xticks_rotation='vertical')
    summary_writer.add_figure(f'confusion_matrix after neuropose using {class_name}', plt.gcf())

def linear_evaluation(train_loader, test_loader, model, classes, writer, device, plot=False):
    classifiers = make_linear_classifiers(cfg)
    #  get features
    train_features, train_labels = get_features(train_loader, model, device=device)
    test_features, test_labels = get_features(test_loader, model, device=device)

    #  train classifiers
    for i, (class_name, classifier) in enumerate(classifiers.items()):
        print(f"Training {class_name}")
        classifier.fit(train_features, train_labels)
        pred = classifier.predict(test_features)
        score_linear(class_name, pred, test_labels, classes, writer)
    
    #  clustering 
    clustering = make_clustering(cfg)
    for i, (class_name, cluster) in enumerate(clustering.items()):
        print(f"Training {class_name}")
        # pca = PCA(n_components=2)
        x = np.concatenate((train_features, test_features), axis=0)
        # x = pca.fit_transform(x)
        # print number of null values
        print(f"Number of null values: {np.sum(np.isnan(x))}")
        clusters = cluster.fit_predict(x)

        test_clusters = clusters[len(train_features):]
        # test_clusters = np.random.randint(0, 10, len(test_features))

        labels = np.zeros_like(test_clusters)
        for i in range(len(classes)-1):
            mask = (clusters[:len(train_features)] == i)
            if np.sum(mask) == 0:
                print(f"Skipping label {i} as it has no data in the training set")
                continue
            labels[test_clusters == i] = mode(train_labels[mask], axis=0)[0]

        score_linear(class_name, labels, test_labels, classes, writer)
        

def main(cfg):

    # setup seed
    setup_seed(cfg.SEED)

    # setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build data loaders
    dataloaders = build_dataloaders(cfg, pretrain=False)
    classes = dataloaders['train'].dataset.dataset.gesture_mapping_class
    num_classes = len(classes)

    # build model
    model = NeuroPose(output_shape=(cfg.DATA.SEGMENT_LENGTH, len(cfg.DATA.LABEL_COLUMNS)))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    criterion = NeuroLoss(metric=cfg.SOLVER.METRIC, keypoints=cfg.DATA.LABEL_COLUMNS)

    # setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(cfg.SOLVER.LOG_DIR, 'neuropose/tb_logs'))

    # train model
    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        train(dataloaders['train'], model, optimizer, criterion, epoch, writer, device)
        validate(dataloaders['val'], model, criterion, epoch, writer, device, stage='val')
        

    # test model
    validate(dataloaders['test'], model, criterion, epoch, writer, device, stage='test')
    linear_evaluation(dataloaders['train'], dataloaders['test'], model, classes, writer, device)
    # close tensorboard
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neuropose model")
    parser.add_argument("--config", default="config.yaml", type=str, help="path to config file")
    parser.add_argument("--opts", nargs="*", default=[], help="modify config options using the command-line")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    cfg.SOLVER.LOG_DIR = os.path.join(cfg.SOLVER.LOG_DIR, cfg.DATA.EXP_SETUP)
    os.makedirs(cfg.SOLVER.LOG_DIR, exist_ok=True)

    main(cfg)