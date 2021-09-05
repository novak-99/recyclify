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

class LoadImg(Dataset):
    def __init__(self, img, transform=None):
        self.img = img
        self.transforms = transform
    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        img = self.img
        if len(img.shape) == 2: 
            img = gray2rgb(img) # The model can't work with grayscale images. 
        elif img.shape[2] == 4:
            img = rgba2rgb(img) # The model can't work with images w/ an alpha channel.

        if self.transforms: # apply transforms, if any
            img = self.transforms(img.copy())

        return img, [[]]


def class_detector(y_hat):
    if torch.argmax(y_hat).item() == 0:
        return "Paper"
    elif torch.argmax(y_hat).item() == 1:
        return "Plastic"
    elif torch.argmax(y_hat).item() == 2:
        return "Cans"
    elif torch.argmax(y_hat).item() == 3:
        return "Glassware"
    else:
        return "Misc"

transform = torchvision.transforms.Compose([transforms.ToTensor(), transforms.Resize((224,224)),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                            transforms.RandomHorizontalFlip(0.5), transforms.RandomVerticalFlip(0.5),
                                            transforms.RandomCrop(size=224, padding=4, padding_mode='reflect')])

model = torch.load("recycle-class.pth", map_location=torch.device('cpu'))

import cv2

cap = cv2.VideoCapture(0)


while True:
    ret, frame = cap.read()
    
    width, height = int(cap.get(3)), int(cap.get(4))
        
    ds = LoadImg(frame, transform)
    dl = torch.utils.data.DataLoader(dataset=ds, batch_size=1)
    
    y_hat = torch.tensor([[]])
    for images, labels in dl:
        y_hat = model(images)
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    frame = cv2.putText(frame, class_detector(y_hat), (200, height-10), font,4, (255,255,255), 5, cv2.LINE_AA)
    print(class_detector(y_hat))
    cv2.imshow("Recycle Detection Example", frame)
    if cv2.waitKey(1) == ord('q'):
        break
        
        
cap.release()
cv2.destroyAllWindows()