import torch
from torch.utils.data import DataLoader

#code for creating a dataset(random)
def random_dataset():
    img_train = torch.rand(size=(100,3,128,255))
    audio_train = torch.rand(size=(100, 1, 257, 199))
    data = []
    for i in range(100):
        data.append((img_train[i], audio_train[i]))
    return DataLoader(data, batch_size=10, shuffle=True)

#code for creating a dataset from the directory which contains the frames and audio
def create_dataset():
    print("from the directory which consits of dataset")

train_loader = random_dataset()
print(len(train_loader))
