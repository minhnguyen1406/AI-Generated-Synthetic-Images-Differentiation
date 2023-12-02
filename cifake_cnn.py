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
import torch.utils.data
from PIL import Image
import matplotlib
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
#Download CIFAKE Dataset (storing in my google drive, which is open to anyone with the link)
!gdown 1I2EfjlbiZ1sAk33AbGNhvPZROnQVt52U
!unzip archive.zip -d CIFAKE/


class CIFAKE(torch.utils.data.Dataset):

  # Initialize the class e.g. load files, preprocess, etc.
  def __init__(self, split = 'train', transform = None):

    self.categories = ['FAKE', 'REAL']
    self.category2index = {category: idx for (idx, category) in enumerate(self.categories)}
    self.transform = transform

    # Compile a list of images and corresponding labels.
    self.imagepaths = []
    self.labels = []
    for category in self.categories:
      category_directory = 'CIFAKE/' + split +'/'+ category
      category_imagenames = os.listdir(category_directory)
      self.imagepaths += [os.path.join(category_directory, imagename)
                          for imagename in category_imagenames]
      self.labels += [self.category2index[category]] * len(category_imagenames)

    # Sort imagepaths alphabetically and labels accordingly.
    sorted_pairs = sorted(zip(self.imagepaths, self.labels), key = lambda x: x[0])
    self.imagepaths, self.labels = zip(*sorted_pairs)


  # Return a sample (x, y) as a tuple e.g. (image, label)
  def __getitem__(self, index):
    image = Image.open(self.imagepaths[index]).convert('RGB')
    if self.transform:
      image = self.transform(image)
    return image, self.labels[index]

  # Return the total number of samples.
  def __len__(self):
    return len(self.imagepaths)


#This is just a test to see a plot a sample image from the Dataset
#we can also pass in an optional transform if we plan to do any sort of Data Augmentation ahead of training
trainset = CIFAKE(split = 'train')
print(len(trainset))
image_index = 70000 


print('This dataset has {0} training images'.format(len(trainset)))
img, label = trainset[image_index]  # Returns image and label.

print('Image {0} is {1}'.format(image_index, trainset.categories[label]))
print('Image size is {0}x{1}'.format(img.height, img.width))

# Show the image.
# Added bilinear interpolation just to smoothen the image (totally optional)
plt.figure()
plt.imshow(img, interpolation='bilinear')
plt.grid(False)
plt.axis('off')
plt.show()

"""Creating the network"""
class CNN(nn.Module):
    def __init__(self, image_size=32):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32 * (image_size // 4) * (image_size // 4), 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Image size in the dataset is 32
model = CNN(image_size=32)

if cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=lr)

"""Train"""
#TODO: Train: forward, loss calculation, optimizer, accuracy calculation, etc. - Steve
eps = 1e-13
def train(epoch):
   model.train()
   criterion = nn.BCELoss() # using binary loss function
   
   for batch_idx, (data, target) in enumerate(train_loader):
        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(torch.log(output+eps), target)
        loss.backward()
        optimizer.step()

        if batch_idx % logging_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item() )
            )

        n_iter = (epoch - 1) * len(train_loader) + batch_idx + 1
        writer.add_scalar('train/loss', loss.data.item(), n_iter)

    # Not sure if we need all of this. Looks like its writing to a histogram.
    # Log model parameters to TensorBoard at every epoch
    for name, param in model.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram(
            f'{layer}/{attr}',
            param.clone().cpu().data.numpy(),
            n_iter)

"""Test"""
#TODO: Test: loss calculation, optimizer, accuracy , etc. - Alex
def test(epoch):
    model.eval()  # Set the model to evaluation mode

    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0

    with torch.no_grad():  # Disable gradient computation during testing
        for data, target in test_loader:
            
            # data = data.view(-1, 28, 28)
            output = model(data)  # Forward pass

            test_loss += criterion(output, target).item()

            # Get the index of the max log-probability as the predicted label
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(f'Test set: Average loss: {test_loss:.4f}')
    
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Testing Accuracy: {accuracy:.2f}%')
    
    # writer.add_scalar("Testing loss", test_loss, epoch)
    writer.add_scalar("Testing Accuracy", accuracy, epoch)
    return accuracy



