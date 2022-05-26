import argparse
import datetime
import os

import torch
from torch import nn
from torch import optim

from dataloaders import cifar, stl10
from models.vgg import vgg16
from utils.general import init_seeds, draw_lr, draw_acc_and_loss, get_current_time
from utils.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=str, default='0')

logger = Logger()


class CFG():
    best_acc = 0
    best_model_path = None
    result_path = None

    train_acc_list = []
    test_acc_list = []

    train_loss_list = []
    test_loss_list = []

    lr_list = []


def train(dataloader, model, loss_fn, optimizer, cfg, device, print_step=10):
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
    cfg.train_loss_list.append(train_loss)
    cfg.train_acc_list.append(correct)
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
    cfg.test_loss_list.append(test_loss)
    cfg.test_acc_list.append(correct)

    is_best = (correct > cfg.best_acc)
    if is_best:
        cfg.best_acc = correct
        logger.info(f"[FEAT] update best acc: {cfg.best_acc:.4f}")
        model_name=f"best-model-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}-acc{cfg.best_acc:.4f}.pth"
        model_state = {
            'model': model.state_dict(),
            'acc': cfg.best_acc
        }
        update_best_model(cfg, model_state, model_name)
    
    logger.info(f"Test Error: Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")


def update_best_model(cfg, model_state, model_name):
    r"""
    更新权重文件

    Args:
        cfg: must have `result_path` and `best_model_path` attributes
    """
    cp_path = os.path.join(cfg.result_path, model_name)

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

    # result_path
    cfg.result_path = os.path.join(os.getcwd(), "work_dir", get_current_time())
    if not os.path.exists(cfg.result_path):
        os.makedirs(cfg.result_path)
    logger.info(f"result_path: {cfg.result_path}")

    # Get cpu or gpu device for training.
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[INFO] using {device} device")

    batch_size = 128
    epochs = 200

    # Create data loaders.
    train_dataloader = cifar.get10(batch_size, "E:/dataset", train=True, val=False)
    test_dataloader = cifar.get10(batch_size, "E:/dataset", train=False, val=True)

    model = vgg16().to(device)  # 79.18
    # model = vgg_spp16().to(device)  # [1, 2, 4]==>79.29, 79.76%, [1, 2, 4, 5]==>78.54, [1, 2, 3, 4]==>79.12%

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

    for epoch in range(epochs):
        logger.info(f"{'-' * 20} epoch {epoch+1} {'-' * 20}")
        cur_lr = float(optimizer.state_dict()['param_groups'][0]['lr'])
        cfg.lr_list.append(cur_lr)
        logger.info(f"[INFO] lr is: {cur_lr}")
        train(train_dataloader, model, loss_fn, optimizer, cfg, device, 100)
        test(test_dataloader, model, loss_fn, cfg, device, 100)
        scheduler.step()

        draw_lr(cfg.result_path, cfg.lr_list)
        draw_acc_and_loss(
            cfg.train_loss_list, cfg.test_loss_list, 
            cfg.train_acc_list, cfg.test_acc_list,
            cfg.result_path)
        
    logger.info("Done!")

if __name__ == "__main__":
    main()