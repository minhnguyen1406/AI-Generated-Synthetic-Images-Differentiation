# -*- coding: utf-8 -*-
"""Assignment_2_Part_2_RNN_MNIST_vp1.ipynb
Overall structure:

1) Set Pytorch metada
- seed
- tensorflow output
- whether to transfer to gpu (cuda)

2) Import data
- download data
- create data loaders with batchsie, transforms, scaling

3) Define Model architecture, loss and optimizer

4) Define Test and Training loop
    - Train:
        a. get next batch
        b. forward pass through model
        c. calculate loss
        d. backward pass from loss (calculates the gradient for each parameter)
        e. optimizer: performs weight updates

5) Perform Training over multiple epochs:
    Each epoch:
    - call train loop
    - call test loop

# Step 1: Pytorch and Training Metadata
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import gc
from pathlib import Path
import matplotlib.pyplot as plt

batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.0001
try_cuda = True
seed = 1000
logging_interval = 10 # how many batches to wait before logging
logging_dir = None

INPUT_SIZE = 28

# 1) setting up the logging

datetime_str = datetime.now().strftime('%b%d_%H-%M-%S')

if logging_dir is None:
    runs_dir = Path("./") / Path(f"runs/")
    runs_dir.mkdir(exist_ok = True)

    logging_dir = runs_dir / Path(f"{datetime_str}")

    logging_dir.mkdir(exist_ok = True)
    logging_dir = str(logging_dir.absolute())

writer = SummaryWriter(log_dir=logging_dir)

#deciding whether to send to the cpu or not if available
if torch.cuda.is_available() and try_cuda:
    cuda = True
    torch.cuda.manual_seed(seed)
else:
    cuda = False
    torch.manual_seed(seed)

"""# Step 2: Data Setup"""

# Setting up data
transform = transforms.Compose([transforms.ToTensor()])


train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# plot one example
print(train_dataset.train_data.size())     # (60000, 28, 28)
print(train_dataset.train_labels.size())   # (60000)
plt.imshow(train_dataset.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_dataset.train_labels[0])
plt.show()

"""# Step 3: Creating the Model"""

class Net(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        #self.rnn = nn.RNN(INPUT_SIZE, self.hidden_size)
        #self.rnn = nn.LSTM(INPUT_SIZE, self.hidden_size)
        self.rnn = nn.GRU(INPUT_SIZE, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)

        #r_out, hidden = self.rnn(x, None)   # None represents zero initial hidden state
        #r_out, (h_n, c_n) = self.rnn(x, None)
        r_out, h_n = self.rnn(x, None)
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

model = Net(128, 28)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

def calculate_accuracy(output, target):
    pred = torch.argmax(output, 1)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct / target.shape[0]

"""# Step 4: Train/Test"""

# Defining the test and  loops

def train(epoch):
    model.train()
    running_loss = 0
    running_accuracy = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()

        data = data.view(-1, 28, 28)

        optimizer.zero_grad()
        output = model(data) # forward
        loss = criterion(output, target)
        
        loss.backward()  # backward
        optimizer.step()
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(output, target)
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
        writer.add_histogram('{}/{}'.format(layer, attr), param.clone().cpu().data.numpy())
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

        data = data.view(-1, 28, 28)
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(output, target)

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

writer.close()

# Commented out IPython magic to ensure Python compatibility.
"""
#https://stackoverflow.com/questions/55970686/tensorboard-not-found-as-magic-function-in-jupyter

#seems to be working in firefox when not working in Google Chrome when running in Colab
#https://stackoverflow.com/questions/64218755/getting-error-403-in-google-colab-with-tensorboard-with-firefox


# %load_ext tensorboard
# %tensorboard --logdir [dir]

"""