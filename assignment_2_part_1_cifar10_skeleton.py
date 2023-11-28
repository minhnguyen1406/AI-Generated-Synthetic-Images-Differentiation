# -*- coding: utf-8 -*-
"""Assignment_2_Part_1_Cifar10_vp1.ipynb

Purpose: Implement image classsification nn the cifar10
dataset using a pytorch implementation of a CNN architecture (LeNet5)

Pseudocode:
1) Set Pytorch metada
- seed
- tensorboard output (logging)
- whether to transfer to gpu (cuda)

2) Import the data
- download the data
- create the pytorch datasets
    scaling
- create pytorch dataloaders
    transforms
    batch size

3) Define the model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates
        f. Calculate accuracy, other stats
    - Test:
        a. Calculate loss, accuracy, other stats

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop




"""

# Step 1: Pytorch and Training Metadata

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

# Architecture
num_classes = 10

# otherum
logging_interval = 10  # how many batches to wait before logging
logging_dir = None
grayscale = True

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok=True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok=True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

# deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# downloading the cifar10 dataset
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def check_data_loader_dim(loader):
    # Checking the dataset
    for images, labels in loader:
        print('Image batch dimensions:', images.shape)
        print('Image label dimensions:', labels.shape)
        break


check_data_loader_dim(train_loader)
check_data_loader_dim(test_loader)

"""# 3) Creating the Model"""

layer_1_n_filters = 32
layer_2_n_filters = 64
fc_1_n_nodes = 7 * 7 * 64
fc_2_n_nodes = 1024
padding = "same"
kernel_size = 5
verbose = False

# calculating the side length of the final activation maps
final_length = 5

if verbose:
    print(f"final_length = {final_length}")


class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, layer_1_n_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(layer_1_n_filters, layer_2_n_filters, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(final_length * final_length * layer_2_n_filters * in_channels, fc_1_n_nodes),
            nn.Tanh(),
            nn.Linear(fc_1_n_nodes, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


model = LeNet5(num_classes=num_classes, grayscale=grayscale)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)


def calculate_accuracy(output, target):
    pred = torch.argmax(output, 1)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / target.shape[0]


"""# Step 4: Train/Test Loop"""


# Defining the test and training loops
def train(epoch):
    model.train()
    running_loss = 0
    running_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        logits, probas = model(data)  # forward
        loss = criterion(logits, target)
        loss.backward()  # backward
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(logits, target)
        if batch_idx % logging_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item())
            )
            n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
            writer.add_scalar('train/loss', loss.item(), n_iter)

    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy(), n_iter)
    train_loss = running_loss / len(train_loader)
    train_accuracy = running_accuracy / len(train_loader)
    return train_loss, train_accuracy


def test(epoch):
    model.eval()
    running_loss = 0
    running_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()

        logits, probas = model(data)
        loss = criterion(logits, target)
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(logits, target)

    test_loss = running_loss / len(test_loader)
    test_accuracy = running_accuracy / len(test_loader)


    n_iter = epoch * len(test_loader)
    writer.add_scalar('test/loss', test_loss, n_iter)
    writer.add_scalar('test/accuracy', test_accuracy, n_iter)
    return test_loss, test_accuracy

def plot():
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []
    for epoch in range(1, epochs + 1):
        print(f"Model is using {'cuda' if cuda else 'cpu'}")
        train_loss, train_accuracy = train(epoch)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = test(epoch)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        print(f"Training: LOSS: {train_loss} | ACCURACY: {train_accuracy}\n")
        print(f"Test: LOSS: {test_loss} | ACCURACY: {test_accuracy}\n")

        # CLEANUP
        gc.collect()
        torch.cuda.empty_cache()


    # Create a subplot of 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plotting the losses on the first subplot
    ax1.plot(train_losses, color='red', label='Train Loss')
    ax1.plot(test_losses, color='orange', label='Test Loss')
    ax1.set_title('Training and Test Losses')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # Plotting the accuracies on the second subplot
    ax2.plot(train_accuracies, color='green', label='Train Accuracy')
    ax2.plot(test_accuracies, color='blue', label='Test Accuracy')
    ax2.set_title('Training and Test Accuracies')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)

    # Display the plots
    plt.tight_layout()
    plt.show()

plot()
def visualize_filters(layer_weights):
    # Normalize for visualization
    min_val = torch.min(layer_weights)
    layer_weights -= min_val
    max_val = torch.max(layer_weights)
    layer_weights /= max_val

    # Create a grid of filters
    filter_grid = torchvision.utils.make_grid(layer_weights, normalize=True)

    plt.imshow(filter_grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    plt.show()

# Accessing the first convolutional layer's weights
conv1_weights = model.features[0].weight.data.clone()

visualize_filters(conv1_weights)
def hook_fn(module, input, output):
    mean = output.data.mean().item()
    std = output.data.std().item()
    print(f"Mean: {mean:.4f} | Std: {std:.4f}")

# Hook into the convolutional layers
hooks = []
for layer in model.features:
    if isinstance(layer, nn.Conv2d):
        hook = layer.register_forward_hook(hook_fn)
        hooks.append(hook)

# Pass a test image through the model to trigger the hooks
test_images, _ = next(iter(test_loader))
if cuda:
    test_images = test_images.cuda()
_ = model(test_images)

# Remove hooks
for hook in hooks:
    hook.remove()

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""
