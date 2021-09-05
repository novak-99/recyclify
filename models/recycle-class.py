import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision 
import torchvision.transforms as transforms

import os
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.color import rgba2rgb, gray2rgb

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

# Hyperparameters
lr = 0.01
max_epoch = 10
batch_size = 32

class LoadData(Dataset):
  def __init__(self, root_dir, csv_file, transforms):
    self.root_dir = root_dir
    self.csv_file = pd.read_csv(csv_file, header=None)
    self.n_class = len(self.csv_file.iloc[0]) - 1 # Disregard 0th index 
    self.transforms = transforms

  def __len__(self):
    return len(self.csv_file)

  def __getitem__(self, index):
    img_dir = os.path.join(self.root_dir, self.csv_file.iloc[index, 0]) # 0 is our img_dir index
    img = io.imread(img_dir)

    if len(img.shape) == 2: 
      img = gray2rgb(img) # The model can't work with grayscale images. 
    elif img.shape[2] == 4:
      img = rgba2rgb(img) # The model can't work with images w/ an alpha channel.

    label = np.empty(self.n_class)
    for i in range(self.n_class):
      label[i] = self.csv_file.iloc[index, i+1]

    label = label.astype(np.float32) # to make it compatible with our model

    if self.transforms: # apply transforms, if any
      img = self.transforms(img.copy())

    return img, label

transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
                                            transforms.RandomCrop(size=224, padding=4, padding_mode='reflect')])

data_set = LoadData("drive/MyDrive", "labels.csv", transform)
train_set, test_set = torch.utils.data.random_split(data_set,  [1533, 400])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

model = torchvision.models.resnet18(pretrained=True)

for param in model.parameters():
  param.requires_grad = False


model.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=5),
          )

model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.CrossEntropyLoss()

def eval(model, loader):
  model.eval()
  n_c = 0
  n_s = 0
  for images, labels in loader:
    images, labels = images.to(device).float(), labels.to(device)
    y_hat = model(images)
    n_c += torch.sum(y_hat.argmax(dim=1) == labels.argmax(dim=1)).item()
    n_s += labels.size()[0]

  return n_c/n_s


for epoch in range(max_epoch):
  model.train()
  for images, labels in train_loader: 
    images, labels = images.to(device).float(), labels.to(device)
    y_hat = model(images)
    l = loss(y_hat, labels.argmax(dim=1))
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

  print(f"EPOCH: {epoch+1}, TRAIN ACC: {eval(model, train_loader)}, TEST ACC: {eval(model, test_loader)}")

torch.save(model, "recycle-class.pth")