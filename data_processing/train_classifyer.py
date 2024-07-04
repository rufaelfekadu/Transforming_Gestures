
import xgboost as xgb
from tqdm import tqdm
from lazypredict.Supervised import LazyClassifier
from hpe.config import cfg, dump_cfg_to_yaml
from hpe.models import EmgNet, build_model, build_optimiser
from hpe.data import build_dataloaders_for_classifier
from hpe.loss import build_loss
from hpe.utils.misc import set_seed, setup_logger, AverageMeter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
from contextlib import redirect_stdout
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import datetime
# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self._initialize_weights()
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
def main_NN(cfg):
    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup logger
    logger = setup_logger(cfg)

    # build dataset
    dataloaders = build_dataloaders_for_classifier(cfg)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']

    input_size = next(iter(train_loader))[0].shape[1]
    # num_classes = len(np.unique(next(iter(train_loader))[1].numpy()))
    num_classes = 11#len(np.unique(next(iter(train_loader))[1].numpy()))

    # Initialize the MLP model
    model = MLP(input_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.float().to(device), targets.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Evaluate the model on the validation set
    model.eval()
    val_targets_list = []
    val_predictions_list = []
    with torch.no_grad():
        for val_inputs, val_targets in val_loader:
            val_inputs, val_targets = val_inputs.float().to(device), val_targets.long().to(device)
            val_outputs = model(val_inputs)
            _, val_predictions = torch.max(val_outputs, 1)
            val_targets_list.append(val_targets.cpu().numpy())
            val_predictions_list.append(val_predictions.cpu().numpy())

    val_targets = np.concatenate(val_targets_list)
    val_predictions = np.concatenate(val_predictions_list)
    val_accuracy = accuracy_score(val_targets, val_predictions)
    print(f"Validation Accuracy: {val_accuracy}")

    # Evaluate the model on the test set
    test_targets_list = []
    test_predictions_list = []
    with torch.no_grad():
        for test_inputs, test_targets in test_loader:
            test_inputs, test_targets = test_inputs.float().to(device), test_targets.long().to(device)
            test_outputs = model(test_inputs)
            _, test_predictions = torch.max(test_outputs, 1)
            test_targets_list.append(test_targets.cpu().numpy())
            test_predictions_list.append(test_predictions.cpu().numpy())

    test_targets = np.concatenate(test_targets_list)
    test_predictions = np.concatenate(test_predictions_list)

    # Calculate metrics for the test set
    test_accuracy = accuracy_score(test_targets, test_predictions)
    precision = precision_score(test_targets, test_predictions, average='weighted')
    recall = recall_score(test_targets, test_predictions, average='weighted')
    f1 = f1_score(test_targets, test_predictions, average='weighted')
    conf_matrix = confusion_matrix(test_targets, test_predictions)

    # Print metrics for the test set
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)

def main_classifiers(cfg):
    # setup device
    device = "cpu"

    #  setup logger
    logger = setup_logger(cfg)

    # build dataset
    dataloaders = build_dataloaders_for_classifier(cfg)
    train_features, train_labels = extract_data_from_dataloader(dataloaders['train'])
    val_features, val_labels = extract_data_from_dataloader(dataloaders['val'])
    # bin_val_labels = np.zeros((val_labels.shape[0],len(np.unique(val_labels))))
    # bin_val_labels[np.arange(val_labels.shape[0]),val_labels]=1
    # bin_train_labels = np.zeros((train_labels.shape[0], len(np.unique(train_labels))))
    # bin_train_labels[np.arange(train_labels.shape[0]), train_labels] = 1
    clff = LazyClassifier(verbose=2, ignore_warnings=False, custom_metric=None)
    models, predictions = clff.fit(train_features,val_features, train_labels, val_labels)
    print(models)
    # Capture the output
    # with open('lazypredict_log.txt', 'w') as f:
    #     with redirect_stdout(f):

    # Print the output from the file if needed
    # with open('lazypredict_log.txt', 'r') as f:
    #     print(f.read())

    model = train_xgboost_classifier(train_features, train_labels, val_features, val_labels)
    # Evaluate the model on the test set
    test_features , test_labels = extract_data_from_dataloader(dataloaders['test'])


    test_predictions = model.predict(test_features)
    # Initialize LazyClassifier

    # Calculate metrics
    accuracy = accuracy_score(test_labels, test_predictions)
    precision = precision_score(test_labels, test_predictions, average='weighted')
    recall = recall_score(test_labels, test_predictions, average='weighted')
    f1 = f1_score(test_labels, test_predictions, average='weighted')
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print("Confusion Matrix:")
    print(conf_matrix)
    # Save the metrics to a text file with a timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"metrics_{current_time}.txt"
    with open(filename, 'w') as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1-score: {f1}\n")
        file.write("Confusion Matrix:\n")
        for row in conf_matrix:
            file.write(f"{row}\n")

    print(f"Data saved in file: {filename}")

# Function to build dataloaders
def extract_data_from_dataloader(dataloader):
    features, labels = [], []
    print("extracting data:")
    for data in tqdm(dataloader):
        inputs, targets = data
        features.append(inputs.numpy())
        labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

# Function to train an XGBoost classifier
def train_xgboost_classifier(train_features, train_labels, val_features, val_labels):
    model = xgb.XGBClassifier(objective='multi:softprob', num_class=len(np.unique(train_labels)), eval_metric='mlogloss', tree_method='gpu_hist', gpu_id=0)
    eval_set = [(train_features, train_labels), (val_features, val_labels)]
    model.fit(train_features, train_labels, eval_set=eval_set, verbose=True)
    return model

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finger gesture tracking decoder (Angle to pose)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # cfg.SOLVER.SAVE_DIR = os.path.join(cfg.SOLVER.SAVE_DIR, cfg.DATA.EXP_SETUP)
    # os.makedirs(cfg.SOLVER.SAVE_DIR, exist_ok=True)

    #  set seed
    set_seed(cfg.SEED)

    main_classifiers(cfg)
    # main_NN(cfg)

