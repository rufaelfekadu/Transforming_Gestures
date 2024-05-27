import torch
import random
import numpy as np

from hpe.model.classifier import SimpleMLP, SimpleCNN, SimpleRNN, make_classifier
from hpe.config import cfg
from hpe.data.emgdataset import build_dataloaders

from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def compute_topk(outputs, labels, k):
    _, topk_predictions = outputs.topk(k, dim=1)
    correct = topk_predictions.eq(labels.view(-1, 1).expand_as(topk_predictions))
    return correct.sum().item()


def train(train_loader, model, optimizer, criterion, epoch, writer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(train_loader, 0):
        _,_,_,_,inputs, _, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total += labels.size(0)
        correct += compute_topk(outputs, labels, 1)
    writer.add_scalar('training_loss', running_loss / len(train_loader), epoch)
    writer.add_scalar('training_accuracy', 100. * correct / total, epoch)
    print(f"Train epoch {epoch}: Loss: {running_loss / len(train_loader)}, Accuracy: {100. * correct / total}")

def test(test_loader, model, criterion, epoch, writer, device, plot=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pred = []
    true = []
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            _,_,_,_,inputs, _, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            total += labels.size(0)
            correct += compute_topk(outputs, labels, 1)
            if plot:
                pred.append(outputs.argmax(dim=1).cpu().numpy())
                true.append(labels.cpu().numpy())

    writer.add_scalar('testing_loss', running_loss / len(test_loader), epoch)
    writer.add_scalar('testing_accuracy', 100. * correct / total, epoch)
    if plot:
        # plot confusion matrix
        pred = np.concatenate(pred)
        true = np.concatenate(true)
        cm = ConfusionMatrixDisplay.from_predictions(true, pred, display_labels=test_loader.dataset.dataset.gesture_mapping_class[np.unique(true)])
        cm.plot(xticks_rotation='vertical')
        plt.tight_layout()
        writer.add_figure('confusion_matrix', plt.gcf())
        print(f"Test epoch {epoch}: Loss: {running_loss / len(test_loader)}, Accuracy: {100. * correct / total}")
    else:
        print(f"val epoch {epoch}: Loss: {running_loss / len(test_loader)}, Accuracy: {100. * correct / total}")

def main(cfg):

    #  get data
    dataloaders = build_dataloaders(cfg, pretrain=False)
    classes = dataloaders['train'].dataset.dataset.gesture_mapping_class
    num_classes = len(classes)

    #  setup model
    model = make_classifier(cfg)

    #  setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #  train model
    print('Training model')
    writer = SummaryWriter()

    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        train(dataloaders['train'], model, optimizer, criterion, epoch, writer, device)
        test(dataloaders['val'], model, criterion, epoch, writer, device, plot=False)
    
    #  test model
    print('Testing model')
    test(dataloaders['test'], model, criterion, 0, writer, device, plot=True)
    # test(dataloaders['test_2'], model, criterion, 0, writer, device, plot=True)
