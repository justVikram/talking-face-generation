
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from dataready import crop


class encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.im_conv_1 = nn.Conv2d(3, 64, 4)
        self.im_conv_2 = nn.Conv2d(64, 128, 4)
        self.im_conv_3 = nn.Conv2d(128, 256, 3)
        self.im_conv_4 = nn.Conv2d(256, 512, 3)
        self.im_conv_5 = nn.Conv2d(512, 1024, 3)
        self.fc1 = nn.Linear(5120, 1024)
        self.fc2 = nn.Linear(1024, 256)

        self.au_conv_1 = nn.Conv2d(1, 64, 3)
        self.au_conv_2 = nn.Conv2d(64, 128, 3)
        self.au_conv_3 = nn.Conv2d(128, 256, 3)
        self.au_conv_4 = nn.Conv2d(256, 512, 3)
        self.au_fc1 = nn.Linear(512, 256)

    def forward(self, x, y):
        x = x.view(1, 3, 128, 256)
        y = y.view(1, 1, 257, 199)
        x = F.leaky_relu(self.im_conv_1(x))
        y = F.leaky_relu(self.au_conv_1(y))
        x = F.max_pool2d(x, 2, 2)
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_2(x))
        y = F.leaky_relu(self.au_conv_2(y))
        x = F.max_pool2d(x, 2, 2)
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_3(x))
        y = F.leaky_relu(self.au_conv_3(y))
        x = F.max_pool2d(x, 2, 2)
        y = F.max_pool2d(y, 2)
        x = F.leaky_relu(self.im_conv_4(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.leaky_relu(self.im_conv_5(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5120)

        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        y = F.leaky_relu(self.au_conv_4(y))
        y = F.max_pool2d(y, (28, 21))
        y = y.view(-1, 512)
        y = self.au_fc1(y)
        x = x.view(-1)

        return x, y


