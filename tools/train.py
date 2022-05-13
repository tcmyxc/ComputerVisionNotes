import torch
import os
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

import sys
sys.path.append("/nfs/xwx/DL-digital-image-processing")
from models.vgg import vgg16
from models.vgg_spp import vgg16 as vgg_spp16

from utils.general import init_seeds
from utils.logger import Logger
import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0')

logger = Logger()

class CFG():
    best_acc = 0
    best_model_path = None


def train(dataloader, model, loss_fn, optimizer, device, print_step=10):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % print_step == 0:
            loss, current = loss.item(), batch * len(X)
            logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    logger.info(f"[INFO] Train Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {train_loss:>8f} \n")


def test(dataloader, model, loss_fn, cfg, device, print_step=10):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % print_step == 0:
                loss, current = loss.item(), batch * len(X)
                logger.info(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    test_loss /= num_batches
    correct /= size

    is_best = (correct > cfg.best_acc)
    if is_best:
        cfg.best_acc = correct
        logger.info(f"[FEAT] update best acc: {cfg.best_acc:.4f}")
        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{cfg.best_acc:.4f}.pth"
        model_state = {
            'model': model.state_dict(),
            'acc': cfg.best_acc
        }
        update_best_model("./", cfg, model_state, model_name)
    
    logger.info(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def update_best_model(result_path, cfg, model_state, model_name):
    """更新权重文件"""
    cp_path = os.path.join(result_path, model_name)

    if cfg.best_model_path is not None:
        # remove previous model weights
        os.remove(cfg.best_model_path)

    torch.save(model_state, cp_path)
    cfg.best_model_path = cp_path
    logger.info(f"Saved Best PyTorch Model State to {model_name} \n")


def main():
    args = parser.parse_args()

    init_seeds()

    cfg = CFG()

    # Get cpu or gpu device for training.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[INFO] using {device} device")

    # Download training data from open datasets.
    training_data = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.CIFAR10(
        root="data",
        train=False,
        download=False,
        transform=ToTensor(),
    )

    batch_size = 128
    epochs = 200

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # model = vgg16().to(device)  # 79.18
    model = vgg_spp16().to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    for epoch in range(epochs):
        logger.info(f"{'-' * 20} epoch {epoch+1} {'-' * 20}")
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        logger.info(f"[INFO] lr is: {cur_lr}")
        train(train_dataloader, model, loss_fn, optimizer, device)
        test(test_dataloader, model, loss_fn, cfg, device)
        scheduler.step()
    logger.info("Done!")

if __name__ == "__main__":
    main()