"""
General utils
"""

import os
import random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import datetime


def get_current_time():
    '''get current time'''
    # utc_plus_8_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    utc_plus_8_time = datetime.datetime.now()
    ymd = f"{utc_plus_8_time.year}-{utc_plus_8_time.month:0>2d}-{utc_plus_8_time.day:0>2d}"
    hms = f"{utc_plus_8_time.hour:0>2d}-{utc_plus_8_time.minute:0>2d}-{utc_plus_8_time.second:0>2d}"
    return f"{ymd}_{hms}"


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


def draw_lr(result_path, lr_list: list):
    """绘制lr曲线"""

    np.save(os.path.join(result_path, "model_lr.npy"), np.array(lr_list))

    num_epochs = len(lr_list)

    plt.plot(range(1, num_epochs + 1), lr_list, label='lr')

    plt.title("Learning rate of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Learning rate")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "model_lr.jpg"))
    plt.clf()
    plt.close()


def draw_acc_and_loss(train_loss, test_loss, 
                      train_acc, test_acc, 
                      result_path, filename=None):
    """绘制acc和loss曲线"""
    
    history = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": test_loss,
        "val_acc": test_acc
    }
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    np.save(os.path.join(result_path, "model_acc_loss.npy" if filename is None else f"{filename}.npy"), history)

    num_epochs = len(train_loss)

    plt.plot(range(1, num_epochs + 1), train_loss, "r", label="train loss")
    plt.plot(range(1, num_epochs + 1), test_loss, "b", label="val loss")

    plt.plot(range(1, num_epochs + 1), train_acc, "g", label="train acc")
    plt.plot(range(1, num_epochs + 1), test_acc, "k", label="val acc")

    plt.title("Acc and Loss of each epoch")
    plt.xlabel("Training Epochs")
    plt.ylabel("Acc & Loss")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(result_path, "model_acc_loss.jpg" if filename is None else f"{filename}.jpg"))
    plt.clf()
    plt.close()


if __name__ == '__main__':
    init_seeds()