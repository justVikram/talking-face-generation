import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def create_dataset():
    img_train = torch.rand(size=(2500, 3, 128, 256))
    audio_train = torch.rand(size=(2500, 1, 257, 199))
    data = []
    for i in range(1500):
        data.append((img_train[i], audio_train[i]))
    return DataLoader(data, batch_size=10, shuffle=True)

# for creating a dataset from the directory which contains the frames and audio