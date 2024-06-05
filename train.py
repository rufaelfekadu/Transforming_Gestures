from tg.config import cfg
from tg.models import EmgNet
from tg.data import build_dataloaders
from tg.utils.misc import setup_seed
import argparse
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pytorch_lightning import loggers as pl_loggers
import pytorch_lightning as pl
from tg.models.emgnet import EmgNet
from tg.config import cfg
from tg.data.emgdataset import build_dataloaders
import argparse
import os


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:

        inputs, labels = inputs.to(device), labels.to(device)
        continue
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return epoch_loss, accuracy


def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss function, and optimizer
    model = EmgNet(cfg=cfg).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    # Datasets and dataloaders
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = build_dataloaders(cfg,True)
    val_dataset = build_dataloaders(cfg)

    train_loader = DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(cfg.SOLVER.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{cfg.SOLVER.NUM_EPOCHS}, Train Loss: {train_loss:.4f}, '
              f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

        # Early stopping and checkpoint saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(cfg.SOLVER.LOG_DIR, 'best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= cfg.SOLVER.PATIENCE:
            print("Early stopping triggered.")
            break

    # Load the best model for testing
    model.load_state_dict(torch.load(checkpoint_path))

    # Test the model
    test_loader = DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE,
                             shuffle=False)  # Use the same val_dataset for simplicity
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(["MODEL.FRAME_PATCHES",5])
    #cfg.freeze()

    #  set seed
    setup_seed(cfg.SEED)


    main(cfg)
