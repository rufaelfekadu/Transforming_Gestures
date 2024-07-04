from tqdm import tqdm
from hpe.config import cfg
from hpe.data import build_dataloaders_for_classifier
from hpe.utils.misc import set_seed, setup_logger
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd


# Save the current directory
current_dir = os.getcwd()

# Change to the parent directory
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
os.chdir(parent_dir)
def main(cfg):
    # Create directory with timestamp
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = os.path.join("data_processing",f"{cfg.DATA.EXP_SETUP}_metrics_{current_time}")
    os.makedirs(dir_name, exist_ok=True)
    with open(os.path.join(dir_name, 'config.yaml'), 'w') as f:
        f.write(cfg.dump())
    main_classifiers(cfg,dir_name)
    main_classifiers_shuffeld_data_condition(cfg,dir_name)
def main_classifiers(cfg,dir_name):
    # setup logger
    # logger = setup_logger(cfg)

    # build dataset
    dataloaders,class_mapping = build_dataloaders_for_classifier(cfg,rep=np.arange(10))
    train_features, train_labels = extract_data_from_dataloader(dataloaders['train'])
    val_features, val_labels = extract_data_from_dataloader(dataloaders['val'])
    train_features = np.vstack((train_features,val_features))
    train_labels = np.hstack((train_labels, val_labels))
    # Train an ExtraTreesClassifier
    model = train_extra_trees_classifier(train_features, train_labels)

    # Evaluate the model on the validation set
    # evaluate_model(model, val_features, val_labels, 'Validation',dir_name)

    # Evaluate the model on the test set
    test_features, test_labels = extract_data_from_dataloader(dataloaders['test'])
    evaluate_model(model, test_features, test_labels, 'Test',dir_name,class_mapping)
def remove_repeated_samples(x, y):
    # Concatenate x and y to keep them paired
    combined = np.hstack((x, y.reshape(-1, 1)))

    # Find unique rows in the combined array
    unique_combined, indices = np.unique(combined, axis=0, return_index=True)

    # Sort the indices to maintain the original order
    sorted_indices = np.sort(indices)

    # Split the unique combined array back into x and y
    unique_x = unique_combined[:, :-1]
    unique_y = unique_combined[:, -1]

    return unique_x, unique_y
def main_classifiers_shuffeld_data_condition(cfg,dir_name):
    test_precent =0.95
    val_precent = 0
    # setup logger
    # logger = setup_logger(cfg)

    # build dataset
    dataloaders,class_mapping = build_dataloaders_for_classifier(cfg,rep=np.arange(10))
    train_features, train_labels = extract_data_from_dataloader(dataloaders['train'])
    val_features, val_labels = extract_data_from_dataloader(dataloaders['val'])
    test_features, test_labels = extract_data_from_dataloader(dataloaders['test'])
    features = np.vstack((train_features,val_features,test_features))

    labels = np.hstack((train_labels, val_labels,test_labels))
    features,labels = remove_repeated_samples(features,labels)
    labels_df = pd.DataFrame({'Labels': labels})
    plt.figure(figsize=(12, 6))
    sns.histplot(labels_df, x='Labels', color='blue', label='True Labels', kde=False)
    if class_mapping is not None:
        plt.xticks(ticks=range(len(class_mapping)), labels=class_mapping,rotation=45)
    plt.title(f'Class Distribution')
    plt.legend()
    plt.savefig(os.path.join(dir_name, f'class_distribution_histogram.png'))
    plt.show()
    # np.random.shuffle(labels)
    train_features, val_features, train_labels, val_labels = train_test_split(features, labels,
                                                                              test_size=val_precent + test_precent)
    # Train an ExtraTreesClassifier
    model = train_extra_trees_classifier(train_features, train_labels)

    if val_precent<=0:
        test_features,test_labels=val_features,val_labels
    else:
        val_features,test_features, val_labels,test_labels= train_test_split(val_features,val_labels,test_size=test_precent/(val_precent+test_precent))
        # Evaluate the model on the validation set
        evaluate_model(model, val_features, val_labels, 'Validation_shuffled',dir_name,class_mapping)
    # Evaluate the model on the test set
    evaluate_model(model, test_features, test_labels, 'Test_shuffled',dir_name,class_mapping)








def extract_data_from_dataloader(dataloader):
    features, labels = [], []
    print("Extracting data:")
    for data in tqdm(dataloader):
        inputs, targets = data
        features.append(inputs.numpy())
        labels.append(targets.numpy())
    features = np.concatenate(features)
    labels = np.concatenate(labels)
    return features, labels

def train_extra_trees_classifier(train_features, train_labels):
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(train_features, train_labels)
    return model

def evaluate_model(model, features, labels, dataset_name,dir_name,class_mapping=None):
    predictions = model.predict(features)

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, predictions)

    # Print metrics
    print(f"{dataset_name} Accuracy: {accuracy}")
    print(f"{dataset_name} Precision: {precision}")
    print(f"{dataset_name} Recall: {recall}")
    print(f"{dataset_name} F1-score: {f1}")
    print(f"{dataset_name} Confusion Matrix:")
    print(conf_matrix)


    # Save the metrics to a text file
    filename = os.path.join(dir_name, f'{dataset_name}_log.txt')
    with open(filename, 'w') as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1-score: {f1}\n")
        file.write("Confusion Matrix:\n")
        for row in conf_matrix:
            file.write(f"{row}\n")
    # Dump the configuration to the specified YAML file

    print(f"Data saved in file: {filename}")

    # Plot and save the confusion matrix
    plt.figure(figsize=(10, 7))
    # sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    if class_mapping is not None:
        class_labels = [class_mapping[i] for i in range(len(class_mapping))]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    else:
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{dataset_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(dir_name, f'{dataset_name}_conf_mat.png'))
    plt.close()

    # Plot histograms
    labels_df = pd.DataFrame({'Labels': labels})
    predictions_df = pd.DataFrame({'Predictions': predictions})

    plt.figure(figsize=(12, 6))
    sns.histplot(labels_df, x='Labels', color='blue', label='True Labels', kde=False)
    if class_mapping is not None:
        plt.xticks(ticks=range(len(class_labels)), labels=class_labels,rotation=45)
    sns.histplot(predictions_df, x='Predictions', color='red', label='Predictions', kde=False)
    plt.title(f'{dataset_name} Labels and Predictions Histogram')
    plt.legend()
    plt.savefig(os.path.join(dir_name, f'{dataset_name}_histogram.png'))
    plt.show()

    return accuracy, precision, recall, f1, conf_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Finger gesture tracking decoder (Angle to pose)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--opts', nargs='*', default=[], help='Modify config options using the command-line')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(args.opts)

    set_seed(cfg.SEED)

    main(cfg)
    # main_classifiers_shuffeld_data_condition(cfg)