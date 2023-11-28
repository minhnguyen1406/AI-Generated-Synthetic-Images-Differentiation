import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
from pathlib import Path
import gc
import matplotlib.pyplot as plt

# hyperparameters
batch_size = 128
epochs = 10
lr = 0.001
try_cuda = True
seed = 1000

# otherum
logging_interval = 10  # how many batches to wait before logging
logging_dir = None

"""Logging setup"""
datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')
if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)
    logging_dir = runs_dir / Path(f"{datetime_str}")
    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())
writer = SummaryWriter(log_dir=logging_dir)


"""Cuda GPU check"""
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)


"""Data load & Data preprocessing"""
# downloading the CIFAKE dataset and preprocess if needed


"""Creating the network"""


"""Train"""


"""Test"""



