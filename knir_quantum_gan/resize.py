import itertools
import math
import os
import random
from timeit import default_timer as timer

import cv2 as cv
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pennylane as qml
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn import datasets


class DigitsDataset(Dataset):
    """Pytorch dataloader for the Optical Recognition of Handwritten Digits Data Set"""

    def __init__(self, csv_file, image_size=28, labels=None, transform=None, inter=cv.INTER_NEAREST):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if labels is None:
            labels = [0]
        self.image_size = image_size
        self.csv_file = csv_file
        self.transform = transform
        self.df = self.filter_by_label(labels)
        self.inter = inter

    def filter_by_label(self, label):
        # Use pandas to return a dataframe of only zeros
        df = pd.read_csv(self.csv_file)
        df = df.loc[df['label'].isin(label)]
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = self.df.iloc[idx, :-1] / 16

        image = np.array(image)
        image = image.astype(np.float32).reshape(28, 28)

        if self.transform:
                image = self.transform(image)

        # Return image and label
        return image, 0


image_size=16


images = []

for inter in [torchvision.transforms.InterpolationMode.NEAREST, torchvision.transforms.InterpolationMode.NEAREST_EXACT,
              torchvision.transforms.InterpolationMode.BILINEAR, torchvision.transforms.InterpolationMode.BICUBIC]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size, inter), transforms.Normalize((0.5,), (0.5,))])

    train_set = DigitsDataset(csv_file="./quantum_gans/mnist_test.csv", image_size=image_size, transform=transform, labels=[0,1,2,3,4,5,6,7,8,9], inter=inter)
    for i in range(16):
        ax = plt.subplot(4, 4, i + 1)
        im = train_set[i][0].resize(image_size, image_size)
        plt.imshow(im, cmap="gray_r")
        plt.xticks([])
        plt.yticks([])
    plt.show()
